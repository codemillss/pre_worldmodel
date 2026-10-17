"""Read-only loss-gradient components on one shared K5 prediction graph.

Raw derivatives remain float32. Vector summaries recombine those saved raw
derivatives in float64. Euclidean gradient directions describe local SGD
geometry only; they do not identify an Adam update or a training-run cause.
"""
from __future__ import annotations

import numpy as np
import torch

from .relative_loss_training import relative_scale, relative_sequence_loss
from .training import TrainingError

COMPONENT_ORDER=('one_step','rollout','relative_teacher','relative_rollout','total')
WEIGHTS=np.asarray([1.,.1,.2,.02],np.float64)
NORM_FLOOR=1e-12
ORTHOGONAL_COSINE=1e-6
CLOSURE_MULTIPLIER=64.


class GradientProbeError(TrainingError):
    """The diagnostic cannot preserve its numeric or immutable-state contract."""


def _require(condition,message):
    if not condition:raise GradientProbeError(message)


def direction_pair(first,second):
    """Float64 vector relation; near-zero norms make direction undefined."""
    a,b=np.asarray(first,np.float64),np.asarray(second,np.float64)
    _require(a.shape==b.shape and a.ndim==1 and len(a)>0
             and np.isfinite(a).all() and np.isfinite(b).all(),'Finite matched gradient vectors required')
    na,nb=float(np.linalg.norm(a)),float(np.linalg.norm(b));dot=float(a@b)
    defined=na>NORM_FLOOR and nb>NORM_FLOOR
    cosine=float(dot/(na*nb)) if defined else None
    relation=('UNDEFINED_NEAR_ZERO_NORM' if not defined else
              'NEAR_ORTHOGONAL' if abs(cosine)<=ORTHOGONAL_COSINE else
              'OPPOSED' if cosine<0 else 'ALIGNED')
    return {'first_norm':na,'second_norm':nb,'dot':dot,'cosine':cosine,'relation':relation,
            'second_to_first_norm_ratio':nb/na if na>NORM_FLOOR else None,
            'first_projection_gain_on_second':dot/(na*na) if na>NORM_FLOOR else None,
            'second_projection_gain_on_first':dot/(nb*nb) if nb>NORM_FLOOR else None,
            'first_loss_derivative_along_negative_second':-dot,
            'scope':'Euclidean raw-gradient local SGD geometry only; not an Adam update or causal training attribution'}


def vector_summary(gradients):
    raw=np.asarray(gradients)
    _require(raw.dtype==np.float32 and raw.ndim==2 and raw.shape[0]==5 and raw.shape[1]>0
             and np.isfinite(raw).all(),'Five finite float32 raw gradient vectors required')
    g=raw.astype(np.float64)
    base=g[0]+.1*g[1];aux=.2*(g[2]+.1*g[3]);derived=base+aux;total=g[4]
    residual=total-derived
    weighted_normsum=float(sum(abs(w)*np.linalg.norm(v) for w,v in zip(WEIGHTS,g[:4])))
    bound=float(CLOSURE_MULTIPLIER*np.finfo(np.float32).eps*weighted_normsum+1e-12)
    l2=float(np.linalg.norm(residual));actual_norm=float(np.linalg.norm(total))
    base_total=direction_pair(base,total);aux_total=direction_pair(aux,total)
    summary={'arithmetic':'NumPy float64 recombination of saved raw float32 component derivatives; derived float32 arrays are separate rounded conveniences',
        'norm_floor':NORM_FLOOR,'near_orthogonal_absolute_cosine':ORTHOGONAL_COSINE,
        'raw_component_norms':{name:float(np.linalg.norm(v)) for name,v in zip(COMPONENT_ORDER,g)},
        'derived_norms':{'base':float(np.linalg.norm(base)),'weighted_aux':float(np.linalg.norm(aux)),
                         'total_from_components':float(np.linalg.norm(derived)),'independent_total':actual_norm},
        'base_vs_weighted_aux':direction_pair(base,aux),
        'base_vs_independent_total':base_total,'weighted_aux_vs_independent_total':aux_total,
        'base_projection_gain':base_total['first_projection_gain_on_second'],
        'aux_projection_gain':aux_total['first_projection_gain_on_second'],
        'raw_teacher_free':{'one_step_vs_rollout':direction_pair(g[0],g[1]),
                            'relative_teacher_vs_relative_rollout':direction_pair(g[2],g[3])},
        'closure':{'l2_residual':l2,'max_absolute_residual':float(np.max(np.abs(residual))),
            'relative_l2_residual_descriptive':l2/actual_norm if actual_norm>NORM_FLOOR else None,
            'weighted_component_norm_sum':weighted_normsum,'l2_bound':bound,
            'criterion':'L2(actual_total - float64 weighted component sum) <= 64*eps(float32)*sum(abs(weight_i)*norm64(g_i))+1e-12',
            'pass':bool(l2<=bound),'scope':'Heuristic precision guard for reordered float32 backward accumulation; not a physical or learning-effect threshold'},
        'scope':'Fixed model and fixed batch derivatives. Negative gradient dot suggests local Euclidean interference; no Adam-step, generalization or unique-cause claim'}
    return summary,{'base':base.astype(np.float32),'weighted_aux':aux.astype(np.float32),
                    'total_from_components':derived.astype(np.float32)}


def clip_diagnostic(flat,parameter_order):
    """Replicate norm/coefficient arithmetic without setting .grad or clipping."""
    value=np.asarray(flat)
    _require(value.dtype==np.float32 and value.ndim==1 and np.isfinite(value).all(),
             'Clip diagnosis needs a finite float32 gradient vector')
    norms=[]
    for item in parameter_order:
        block=value[item['start']:item['end']].reshape(item['shape'])
        norms.append(torch.linalg.vector_norm(torch.from_numpy(block.copy()),ord=2))
    _require(len(norms)>0 and parameter_order[-1]['end']==len(value),'Parameter slices do not cover gradient vector')
    norm=torch.linalg.vector_norm(torch.stack(norms),ord=2)
    coefficient=torch.clamp(1./(norm+1e-6),max=1.)
    _require(bool(torch.isfinite(norm)) and bool(torch.isfinite(coefficient)), 'Nonfinite clip diagnosis')
    flat_norm=float(np.linalg.norm(value.astype(np.float64)))
    return {'parameterwise_float32_norm':float(norm),'flat_float64_norm':flat_norm,
        'coefficient':float(coefficient),'maximum_norm':1.,'epsilon':1e-6,
        'parameter_norms_float32':[float(n) for n in norms],
        'would_scale':bool(coefficient<1), 'applied_to_parameters_or_grad_buffers':False,
        'positive_global_scalar_preserves_Euclidean_direction':bool(coefficient>0) if flat_norm>NORM_FLOOR else None,
        'scope':'Norm/coefficient calculation only, no tensor gradients assigned and no clipping or optimizer update performed'}


def parameter_block_summary(gradients,parameter_order):
    """Subsets of parameter-update coordinates, never physical causation."""
    raw=np.asarray(gradients)
    _require(raw.dtype==np.float32 and raw.ndim==2 and raw.shape[0]==5
             and np.isfinite(raw).all(),'Block summaries require five finite float32 vectors')
    g=raw.astype(np.float64);base=g[0]+.1*g[1];aux=.2*(g[2]+.1*g[3]);total=g[4]
    blocks=[];last=0
    for item in parameter_order:
        start,end=item['start'],item['end']
        _require(start==last and end>start and end-start==int(np.prod(item['shape']))
                 and end<=raw.shape[1],'Parameter block slices must exactly cover stored vectors')
        blocks.append({**item,'base_vs_aux':direction_pair(base[start:end],aux[start:end]),
                       'base_vs_total':direction_pair(base[start:end],total[start:end])})
        last=end
    _require(last==raw.shape[1] and len({p['name'] for p in parameter_order})==len(parameter_order),
             'Parameter block names and full coverage required')
    by_name={item['name']:item for item in parameter_order}
    weight,bias=by_name.get('network.4.weight'),by_name.get('network.4.bias')
    scope='Subsets of final-layer parameter-gradient coordinates associated with output rows; not input attribution, observed state response, physical causation or Adam-step evidence'
    applicable=(weight is not None and bias is not None and len(weight['shape'])==2
                and weight['shape'][0]==10 and weight['shape'][1]>0 and bias['shape']==[10])
    if not applicable:
        return blocks,{'status':'NOT_APPLICABLE','groups':{},'scope':scope,
            'reason':'Registered network.4 ten-output Linear weight and bias are absent'}
    width=weight['shape'][1]
    groups={}
    for name,rows in (('object_xy',[0,1]),('yaw',[2]),('object_velocity',[3,4,5]),
                      ('pusher_xy',[6,7]),('pusher_velocity',[8,9])):
        indices=np.concatenate((np.concatenate([weight['start']+r*width+np.arange(width) for r in rows]),
                                bias['start']+np.asarray(rows))).astype(np.int64)
        groups[name]={'output_rows':rows,'parameter_count':len(indices),
            'weight_parameter':'network.4.weight','bias_parameter':'network.4.bias',
            'base_vs_aux':direction_pair(base[indices],aux[indices]),
            'base_vs_total':direction_pair(base[indices],total[indices])}
    return blocks,{'status':'APPLICABLE','groups':groups,'scope':scope,
                  'partition':'All ten final output weight rows and biases exactly once; hidden parameters excluded'}


def gradient_components(model,normalizer,states,actions,mask):
    """Five independent derivatives of one shared teacher+free5 graph.

    The model must already be in eval mode with CPU float32 trainable
    parameters and clean grad buffers. The loss graph still differentiates
    recurrent predictions; eval mode does not disable autograd.
    """
    _require(isinstance(model,torch.nn.Module) and torch.is_grad_enabled(),'Model and enabled autograd required')
    _require(all(not m.training for m in model.modules()),'Probe model must already be in eval mode')
    _require(isinstance(actions,torch.Tensor) and actions.ndim==3 and actions.shape[1]==5,'Registered gradient probe requires K5')
    named=tuple(model.named_parameters());parameters=tuple(p for _,p in named)
    _require(bool(named),'At least one trainable parameter required')
    _require(all(p.requires_grad and p.dtype==torch.float32 and p.device.type=='cpu'
                 and bool(torch.isfinite(p).all()) for p in parameters),
             'Probe parameters must be finite trainable CPU float32')
    _require(all(p.grad is None for p in parameters),'Clean None parameter grad buffers required; probe never zeroes caller gradients')
    before={name:value.detach().clone() for name,value in model.state_dict().items()}
    rng=torch.get_rng_state().clone();modes=[m.training for m in model.modules()]
    order=[];offset=0
    for name,p in named:
        order.append({'name':name,'shape':list(p.shape),'numel':p.numel(),'start':offset,'end':offset+p.numel()})
        offset+=p.numel()
    forward_rows=[]
    handle=model.register_forward_hook(lambda _module,inputs,_output:forward_rows.append(int(inputs[0].shape[0])))
    try:
        terms=relative_sequence_loss(model,normalizer,states,actions,mask,relative_weight=.2,rollout_weight=.1)
        raw=[]
        for index,name in enumerate(COMPONENT_ORDER):
            term=getattr(terms,name)
            _require(term is not None and bool(torch.isfinite(term)),'Every registered raw scalar loss must be finite')
            derivatives=torch.autograd.grad(term,parameters,retain_graph=index<4,create_graph=False,allow_unused=False)
            _require(all(p.dtype==torch.float32 and bool(torch.isfinite(p).all()) for p in derivatives),
                     'Nonfinite or non-float32 component gradient')
            raw.append(torch.cat([p.reshape(-1) for p in derivatives]).detach().numpy().copy())
        gradients=np.stack(raw);summary,derived=vector_summary(gradients)
        _require(summary['closure']['pass'],'Independent total derivative does not match component closure bound')
        summary['parameter_blocks'],summary['output_head_groups']=parameter_block_summary(gradients,order)
        expected_rows=[int(mask.sum())]+[int(mask[:,k].sum()) for k in range(5) if bool(mask[:,k].any())]
        _require(forward_rows==expected_rows,'Shared graph did not use exactly one teacher and active free-step forwards')
        summary['clip_diagnostic']={'base_from_component_sum_float32':clip_diagnostic(derived['base'],order),
                                    'independent_total_float32':clip_diagnostic(gradients[4],order)}
        return {'gradients':gradients,'component_order':list(COMPONENT_ORDER),
            'component_weights':[1.,.1,.2,.02], 'derived_gradients':derived,'parameter_order':order,
            'loss_values':{name:float(getattr(terms,name).detach()) for name in COMPONENT_ORDER},
            'denominators':{'valid_transitions':terms.valid_transitions,'base_scalar_terms':terms.scalar_terms,
                            'relative_scalar_terms':terms.relative_scalar_terms},
            'relative_scale_m':relative_scale(normalizer).tolist(),
            'teacher_xy_errors':terms.teacher_xy_errors.detach().numpy().copy(),
            'free_xy_errors':terms.free_xy_errors.detach().numpy().copy(),
            'vector_summary':summary,'forward_calls':len(forward_rows),'forward_rows':forward_rows,
            'forward_total_rows':sum(forward_rows),'derivative_traversals':5,'parameter_count':offset,
            'optimizer_steps':0,'physics_steps':0,
            'scope':'Read-only component autograd on one K5 graph; no optimizer object created, step/backward/zero_grad called, model selection or new data'}
    finally:
        handle.remove()
        _require(all(p.grad is None for p in parameters),'Probe changed parameter grad buffers')
        _require(all(torch.equal(before[name],value) for name,value in model.state_dict().items()),'Probe changed model state')
        _require(torch.equal(rng,torch.get_rng_state()),'Probe changed Torch RNG')
        _require(modes==[m.training for m in model.modules()],'Probe changed model mode')
