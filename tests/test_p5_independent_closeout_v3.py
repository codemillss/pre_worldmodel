"""Synthetic saved-array adapter checks; no actual run/raw/model/physics use."""
from contextlib import nullcontext
from copy import deepcopy
import ast
import json
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def load(name):
    path=ROOT/'tools'/('p5_final_independent_closeout_'+name+'.py')
    module=types.ModuleType('synthetic_closeout_'+name);module.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module

V2,V3=load('v2'),load('v3')
METRICS=('object_xy_error_m','object_yaw_error_rad_2pi','pusher_xy_error_m',
 'object_linear_velocity_error_m_s','object_angular_velocity_error_rad_s','pusher_velocity_error_m_s')


def fixture(folder,dtype=np.float64,empty=False):
    families=['synthetic-'+str(i) for i in range(12)];arrays={};horizons={}
    for k in (1,5,10):
        prefix=f'K{k}__';truth=np.ones(12,np.bool_)
        arrays[prefix+'truth_valid']=truth;arrays[prefix+'family_id']=np.array(families)
        models={}
        for model in V3.CONTROLLERS[1:]:
            valid=np.zeros(12,np.bool_) if empty else np.ones(12,np.bool_)
            arrays[prefix+model+'__valid']=valid
            value=float(np.array([2 if np.issubdtype(dtype,np.integer) else .25],dtype=dtype)[0])
            statistics={'denominator':0 if empty else 12,
                **{name:None if empty else value for name in ('mean','median','p95','max')}}
            for metric in METRICS:arrays[prefix+model+'__'+metric]=np.full(12,value,dtype=dtype)
            models[model]={'recorded_start_opportunities':12,'truth_eligible_windows':12,
                'predicted_valid_windows':0 if empty else 12,'predicted_invalid_windows':12 if empty else 0,
                'metrics_conditional_on_valid_prediction_and_truth':{m:deepcopy(statistics) for m in METRICS},
                'family_first_means_conditional_on_available_families':{
                    m:{'family_denominator':0 if empty else 12,'mean':None if empty else value} for m in METRICS}}
        horizons[str(k)]={'recorded_start_opportunities':12,'truth_eligible_windows':12,
            'boundary_excluded_windows':0,'full_horizon_windows':12,'raw_invalid_full_windows':0,'models':models}
    summary={'schema':'p5-common-G-fixed-trace-prediction-v1','status':'COMPLETE_DECLARED_RAW_EVALUATION',
        'family_count':12,'family_ids':families,'horizons':horizons,'additional_physics_intervals':0,
        'online_settings_mutated':False,'future_engine_calls':{k:0 for k in V3.ENGINE_GUARDS},
        'inference':{m:{} for m in V3.CONTROLLERS[1:]}}
    path=folder/'fixed_prediction';path.mkdir();(path/'summary.json').write_text(json.dumps(summary))
    class SavedOnly:
        run=folder
        manifest={'families':[{'family_id':f} for f in families]}
        def sealed_stage(self,name,files):
            assert name=='fixed_prediction' and files=={'summary.json','traces.npz'}
        def npz(self,path):return nullcontext(arrays)
    return SavedOnly(),summary,arrays


class ScalarAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.folder=Path(self.temp.name)
    def test_numpy_scalar_maxima_normalized_without_relaxing_close(self):
        for index,dtype in enumerate((np.float64,np.float32,np.int64)):
            with self.subTest(dtype=dtype):
                folder=self.folder/str(index);folder.mkdir();dummy,_,_=fixture(folder,dtype)
                with self.assertRaisesRegex(ValueError,'Prediction metric max: numeric difference'):
                    V2.Audit.fixed_prediction(dummy)
                self.assertEqual(V3.Audit.fixed_prediction(dummy)['models'],7)
                with self.assertRaises(ValueError):V3.close(1.,dtype(1),'direct exact-type guard remains')
    def test_genuine_saved_numeric_difference_is_rejected(self):
        dummy,summary,_=fixture(self.folder)
        summary['horizons']['1']['models']['R']['metrics_conditional_on_valid_prediction_and_truth'][METRICS[0]]['max']+=.01
        (self.folder/'fixed_prediction/summary.json').write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError,'Prediction metric max: numeric difference'):V3.Audit.fixed_prediction(dummy)
    def test_nan_and_infinity_still_rejected(self):
        for value in (np.nan,np.inf,-np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):V3.close(1.,float(max(np.array([value]))),'nonfinite maximum')
        dummy,_,arrays=fixture(self.folder)
        arrays['K1__R__'+METRICS[0]][:]=np.nan
        with self.assertRaises(ValueError):V3.Audit.fixed_prediction(dummy)
    def test_empty_valid_population_none_preserved(self):
        dummy,_,_=fixture(self.folder,empty=True)
        self.assertEqual(V2.Audit.fixed_prediction(dummy),V3.Audit.fixed_prediction(dummy))
        V3.close(None,None,'empty')
        with self.assertRaises(ValueError):V3.close(None,0.,'nonempty mismatch')
    def test_only_intended_arithmetic_ast_change(self):
        before=ast.parse((ROOT/'tools/p5_final_independent_closeout_v2.py').read_text())
        after=ast.parse((ROOT/'tools/p5_final_independent_closeout_v3.py').read_text())
        def method(tree,name):
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Audit')
            return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
        class Adapt(ast.NodeTransformer):
            changes=0
            def visit_Call(self,node):
                self.generic_visit(node)
                if isinstance(node.func,ast.Name) and node.func.id=='max' and len(node.args)==1 \
                        and isinstance(node.args[0],ast.Name) and node.args[0].id=='selected':
                    self.changes+=1;return ast.Call(func=ast.Name(id='float',ctx=ast.Load()),args=[node],keywords=[])
                return node
        transform=Adapt();expected=transform.visit(deepcopy(method(before,'fixed_prediction')))
        self.assertEqual(transform.changes,1)
        self.assertEqual(ast.dump(expected,include_attributes=False),ast.dump(method(after,'fixed_prediction'),include_attributes=False))
        for name in ('close','mean','quantiles','validate_recovery_inputs'):
            first=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==name)
            second=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(first,include_attributes=False),ast.dump(second,include_attributes=False))


if __name__=='__main__':unittest.main()
