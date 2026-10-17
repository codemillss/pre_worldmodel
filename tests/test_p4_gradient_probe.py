"""Analytic read-only gradient probes; no recorded models/data or optimizer."""
from copy import deepcopy
import unittest

import numpy as np
import torch
from torch import nn

from push_align.gradient_probe import (COMPONENT_ORDER, GradientProbeError,
    clip_diagnostic, direction_pair, gradient_components, parameter_block_summary, vector_summary)
from test_p4_relative_loss_training import ConstantDelta, RecurrentToy, batch, synthetic_normalizer


class EarlyHookToy(RecurrentToy):
    def __init__(self):
        super().__init__();self.early_derivatives=[]

    def forward(self,features):
        result=super().forward(features)
        if len(self.outputs)==2:
            result.register_hook(lambda gradient:self.early_derivatives.append(gradient.detach().numpy().copy()))
        return result


class InfDerivative(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value):return value.clone()
    @staticmethod
    def backward(ctx,gradient):return gradient*float('inf')


class NonfiniteGradient(ConstantDelta):
    def forward(self,features):return InfDerivative.apply(super().forward(features))


class GradientProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.normalizer=synthetic_normalizer()

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.previous_threads)

    def make(self,delta=None):return ConstantDelta(delta).eval()

    def test_quadratic_raw_gradients_match_independent_closed_forms(self):
        d=np.zeros(10);d[0]=1.;d[6]=-2.
        result=gradient_components(self.make(d),self.normalizer,*batch((5,),5))
        expected=np.zeros((5,10));expected[0]=.2*d;expected[1]=2.2*d
        expected[2,[0,6]]=[1.5,-1.5];expected[3,[0,6]]=[16.5,-16.5]
        expected[4,[0,6]]=[1.05,-1.47]
        np.testing.assert_allclose(result['gradients'],expected,atol=3e-7,rtol=2e-7)
        self.assertEqual(result['component_order'],list(COMPONENT_ORDER))
        self.assertEqual(result['gradients'].dtype,np.float32)
        self.assertEqual(result['gradients'].shape,(5,10))

    def test_one_teacher_and_five_free_forwards_not_five_forward_rebuilds(self):
        result=gradient_components(self.make(np.arange(10)*.01),self.normalizer,*batch((5,5),5))
        self.assertEqual(result['forward_calls'],6)
        self.assertEqual(result['forward_rows'],[10,2,2,2,2,2])
        self.assertEqual(result['forward_total_rows'],20)
        self.assertEqual(result['derivative_traversals'],5)
        self.assertEqual((result['optimizer_steps'],result['physics_steps']),(0,0))

    def test_prefix_mask_counts_only_recorded_rows_and_keeps_graph(self):
        states,actions,mask=batch((5,2,0),5)
        d=np.zeros(10);d[0]=1.;d[6]=-2.
        result=gradient_components(self.make(d),self.normalizer,states,actions,mask)
        self.assertEqual(result['denominators'],{'valid_transitions':7,'base_scalar_terms':70,'relative_scalar_terms':14})
        self.assertEqual(result['teacher_xy_errors'].shape,(7,4));self.assertEqual(result['free_xy_errors'].shape,(7,4))
        self.assertEqual(result['forward_rows'],[7,2,2,1,1,1])
        np.testing.assert_allclose(result['gradients'][1],2*60/(10*7)*d,atol=8e-7,rtol=2e-7)

    def test_weights_rng_modes_and_none_grad_buffers_preserved(self):
        model=self.make(np.arange(10)*.01);before=deepcopy(model.state_dict())
        rng=torch.get_rng_state().clone();state,action,mask=batch((5,),5)
        state.requires_grad_();action.requires_grad_()
        gradient_components(model,self.normalizer,state,action,mask)
        self.assertFalse(model.training);self.assertIsNone(model.delta.grad)
        self.assertIsNone(state.grad);self.assertIsNone(action.grad)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(value,model.state_dict()[name]) for name,value in before.items()))
        self.assertEqual(len(model._forward_hooks),0)

    def test_grad_buffers_never_silently_zeroed(self):
        model=self.make();model.delta.grad=torch.ones_like(model.delta)
        with self.assertRaisesRegex(GradientProbeError,'Clean None'):
            gradient_components(model,self.normalizer,*batch((5,),5))
        torch.testing.assert_close(model.delta.grad,torch.ones_like(model.delta),atol=0,rtol=0)

    def test_eval_and_enabled_autograd_are_explicit_preconditions(self):
        model=ConstantDelta()
        with self.assertRaisesRegex(GradientProbeError,'eval'):
            gradient_components(model,self.normalizer,*batch((5,),5))
        self.assertTrue(model.training)
        with torch.no_grad(),self.assertRaisesRegex(GradientProbeError,'enabled autograd'):
            gradient_components(model.eval(),self.normalizer,*batch((5,),5))

    def test_k5_and_valid_prefix_required_no_silent_mask_drop(self):
        with self.assertRaisesRegex(GradientProbeError,'K5'):
            gradient_components(self.make(),self.normalizer,*batch((3,),3))
        states,actions,mask=batch((5,),5);mask[0,1]=False
        with self.assertRaisesRegex(ValueError,'prefix'):
            gradient_components(self.make(),self.normalizer,states,actions,mask)
        mask[:]=False
        with self.assertRaisesRegex(ValueError,'valid transition'):
            gradient_components(self.make(),self.normalizer,states,actions,mask)

    def test_nonfinite_valid_truth_is_rejected_without_mutating_model(self):
        model=self.make();states,actions,mask=batch((5,),5);states[0,3,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):
            gradient_components(model,self.normalizer,states,actions,mask)
        self.assertIsNone(model.delta.grad);self.assertFalse(model.training)
        self.assertEqual(len(model._forward_hooks),0)

    def test_nonfinite_derivative_is_rejected_not_filtered(self):
        model=NonfiniteGradient(np.arange(10)*.01).eval()
        with self.assertRaisesRegex(GradientProbeError,'Nonfinite'):
            gradient_components(model,self.normalizer,*batch((5,),5))
        self.assertIsNone(model.delta.grad);self.assertEqual(len(model._forward_hooks),0)

    def test_recursive_free_derivative_reaches_early_output(self):
        model=EarlyHookToy().eval();result=gradient_components(model,self.normalizer,*batch((5,),5))
        k=np.arange(1,6);x=1-.75**k;dx=k*.75**(k-1)
        expected=float(np.sum(x*dx)/10)
        self.assertAlmostEqual(result['gradients'][3,0],expected,places=6)
        self.assertGreaterEqual(len(model.early_derivatives),3)
        self.assertTrue(all(np.isfinite(g).all() and np.abs(g).sum()>0 for g in model.early_derivatives))

    def test_raw_storage_and_float64_summary_derived_definition(self):
        result=gradient_components(self.make(np.arange(10)*.01),self.normalizer,*batch((5,),5))
        raw=result['gradients'].astype(np.float64)
        base=raw[0]+.1*raw[1];aux=.2*(raw[2]+.1*raw[3])
        self.assertEqual(result['vector_summary']['derived_norms']['base'],float(np.linalg.norm(base)))
        self.assertEqual(result['vector_summary']['derived_norms']['weighted_aux'],float(np.linalg.norm(aux)))
        np.testing.assert_array_equal(result['derived_gradients']['base'],base.astype(np.float32))
        self.assertTrue(result['vector_summary']['closure']['pass'])

    def test_zero_norm_directions_are_undefined_not_zero_or_success(self):
        result=gradient_components(self.make(),self.normalizer,*batch((5,),5))
        relation=result['vector_summary']['base_vs_weighted_aux']
        self.assertEqual(relation['relation'],'UNDEFINED_NEAR_ZERO_NORM')
        self.assertIsNone(relation['cosine']);self.assertIsNone(relation['second_to_first_norm_ratio'])
        self.assertIsNone(result['vector_summary']['base_projection_gain'])
        self.assertEqual(result['vector_summary']['closure']['l2_residual'],0.)
        tiny=direction_pair(np.array([1e-13,0]),np.array([1.,0]))
        self.assertIsNone(tiny['cosine'])
        clip=result['vector_summary']['clip_diagnostic']['independent_total_float32']
        self.assertIsNone(clip['positive_global_scalar_preserves_Euclidean_direction'])

    def test_near_orthogonal_retains_signed_dot_and_threshold_relation(self):
        a=np.array([1.,0.]);b=np.array([-5e-7,1.])
        row=direction_pair(a,b)
        self.assertEqual(row['relation'],'NEAR_ORTHOGONAL');self.assertLess(row['dot'],0)
        self.assertGreater(row['first_loss_derivative_along_negative_second'],0)
        self.assertEqual(direction_pair(a,np.array([-1e-4,1.]))['relation'],'OPPOSED')
        self.assertEqual(direction_pair(a,np.array([1e-4,1.]))['relation'],'ALIGNED')

    def test_closure_bound_uses_weighted_normsum_despite_cancellation(self):
        raw=np.zeros((5,2),np.float32);raw[0,0]=1.;raw[1,0]=-10.;raw[4,0]=1e-7
        summary,_=vector_summary(raw);closure=summary['closure']
        expected=64*np.finfo(np.float32).eps*(1.+.1*10)+1e-12
        self.assertAlmostEqual(closure['l2_bound'],expected,places=15)
        self.assertTrue(closure['pass'])
        raw[4,0]=.1
        self.assertFalse(vector_summary(raw)[0]['closure']['pass'])

    def test_clip_diagnosis_matches_torch_norm_formula_without_grad_assignment(self):
        flat=np.array([3.,4.,12.],np.float32)
        order=[{'start':0,'end':2,'shape':[2]},{'start':2,'end':3,'shape':[]}]
        original=flat.copy();row=clip_diagnostic(flat,order)
        expected=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(torch.tensor([3.,4.])),torch.tensor(12.)]))
        self.assertEqual(row['parameterwise_float32_norm'],float(expected))
        self.assertEqual(row['coefficient'],float(torch.clamp(1/(expected+1e-6),max=1)))
        self.assertTrue(row['positive_global_scalar_preserves_Euclidean_direction'])
        self.assertFalse(row['applied_to_parameters_or_grad_buffers']);np.testing.assert_array_equal(flat,original)

    def test_fixed_parameter_order_slices_cover_every_saved_coordinate(self):
        model=nn.Sequential(nn.Linear(11,3),nn.SiLU(),nn.Linear(3,10)).float().eval()
        result=gradient_components(model,self.normalizer,*batch((5,),5))
        self.assertEqual([p['name'] for p in result['parameter_order']],[name for name,_ in model.named_parameters()])
        last=0
        for item in result['parameter_order']:
            self.assertEqual(item['start'],last);last=item['end']
            self.assertEqual(item['numel'],int(np.prod(item['shape'])))
        self.assertEqual(last,result['parameter_count']);self.assertEqual(result['gradients'].shape,(5,last))

    def test_generic_vector_has_parameter_summary_but_no_invented_output_head(self):
        result=gradient_components(self.make(np.arange(10)*.01),self.normalizer,*batch((5,),5))
        summary=result['vector_summary']
        self.assertEqual(len(summary['parameter_blocks']),1)
        self.assertEqual(summary['parameter_blocks'][0]['name'],'delta')
        self.assertEqual(summary['output_head_groups']['status'],'NOT_APPLICABLE')
        self.assertEqual(summary['output_head_groups']['groups'],{})

    def test_synthetic_final_linear_groups_match_independent_selected_rows(self):
        raw=np.zeros((5,30),np.float32);raw[0]=np.arange(30,dtype=np.float32)*.1
        raw[1]=np.arange(30,dtype=np.float32)[::-1]*.2
        raw[2]=np.tile([1.,-1.,2.],10);raw[3]=np.linspace(-2,1,30,dtype=np.float32)
        raw[4]=(raw[0].astype(np.float64)+.1*raw[1]+.2*raw[2]+.02*raw[3]).astype(np.float32)
        order=[{'name':'network.4.weight','shape':[10,2],'numel':20,'start':0,'end':20},
               {'name':'network.4.bias','shape':[10],'numel':10,'start':20,'end':30}]
        blocks,head=parameter_block_summary(raw,order)
        self.assertEqual(len(blocks),2);self.assertEqual(head['status'],'APPLICABLE')
        base=raw[0].astype(np.float64)+.1*raw[1].astype(np.float64)
        aux=.2*(raw[2].astype(np.float64)+.1*raw[3].astype(np.float64))
        self.assertEqual(sum(g['parameter_count'] for g in head['groups'].values()),30)
        for name,rows in [('object_xy',[0,1]),('yaw',[2]),('object_velocity',[3,4,5]),
                          ('pusher_xy',[6,7]),('pusher_velocity',[8,9])]:
            ids=[2*r+c for r in rows for c in range(2)]+[20+r for r in rows]
            expected_dot=float(base[ids]@aux[ids]);nb=float(np.linalg.norm(base[ids]));na=float(np.linalg.norm(aux[ids]))
            got=head['groups'][name]['base_vs_aux']
            self.assertAlmostEqual(got['dot'],expected_dot,places=13)
            self.assertAlmostEqual(got['cosine'],expected_dot/(nb*na),places=13)
            self.assertAlmostEqual(got['second_to_first_norm_ratio'],na/nb,places=13)

    def test_parameter_blocks_reject_gaps_and_head_shape_mismatch_is_not_applicable(self):
        raw=np.ones((5,3),np.float32)
        with self.assertRaisesRegex(GradientProbeError,'slices'):
            parameter_block_summary(raw,[{'name':'bad','shape':[2],'start':1,'end':3}])
        order=[{'name':'network.4.weight','shape':[1,2],'start':0,'end':2},
               {'name':'network.4.bias','shape':[1],'start':2,'end':3}]
        blocks,head=parameter_block_summary(raw,order)
        self.assertEqual(len(blocks),2);self.assertEqual(head['status'],'NOT_APPLICABLE')


if __name__=='__main__':unittest.main()
