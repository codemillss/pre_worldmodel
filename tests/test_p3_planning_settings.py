"""Independent selection-denominator and trace-retention counterexamples."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from push_align.config import ROOT, load_pilot_config
from push_align.planning import PlannerContext, TaskSpec, plan
from push_align.planning_protocol import keyed_planner_rng

sys.path.insert(0,str(ROOT/'scripts'))
import p3_planning_settings as runner


def row(name,loss=1.,terminal='development_budget',status='completed'):
    return {'cell_id':name,'terminal_pose_loss':loss,'termination':terminal,'execution_status':status}


class PlanningSettingsTest(unittest.TestCase):
    def test_missing_or_duplicate_cells_do_not_form_survivor_score(self):
        for rows in ([],[row('A')],[row('A'),row('A')],[row('B'),row('A')]):
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                runner.trial_score(rows,['A','B'])

    def test_hardware_physics_or_geometry_failure_makes_trial_ineligible(self):
        for reason in ('physics_error','geometry_invalid','infrastructure_error','controller_infeasible'):
            score=runner.trial_score([row('A',0.,reason),row('B',1.)],['A','B'])
            self.assertFalse(score['eligible']);self.assertEqual(score['attempted'],2)

    def test_infrastructure_status_alone_cannot_be_hidden(self):
        score=runner.trial_score([row('A',0.,status='infrastructure_error')],['A'])
        self.assertFalse(score['eligible'])

    def test_invalid_metrics_are_not_eligible(self):
        for loss in (None,float('nan'),float('inf'),-1.,True,'0'):
            with self.subTest(loss=loss):
                self.assertFalse(runner.trial_score([row('A',loss)],['A'])['eligible'])

    def test_planner_invalid_is_retained_and_penalized(self):
        score=runner.trial_score([row('A',0.,'planner_invalid'),row('B',1.)],['A','B'])
        self.assertTrue(score['eligible']);self.assertEqual(score['task_failures'],1)
        self.assertEqual(score['mean_terminal_pose_loss'],.5)

    def test_ready_first_then_failure_count_then_pose_loss(self):
        def trial(name,rows):return {'id':name,'score':runner.trial_score(rows,['A'])}
        ready=trial('ready',[row('A',100.,'ready')])
        ongoing=trial('ongoing',[row('A',2.)]);failed=trial('failed',[row('A',0.,'planner_invalid')])
        self.assertIs(runner.select_trial([failed,ongoing,ready]),ready)
        self.assertIs(runner.select_trial([failed,ongoing]),ongoing)

    def test_exact_tie_declared_order_and_close_scores_not_rounded(self):
        left={'id':'first','score':runner.trial_score([row('A',1.)],['A'])}
        right={'id':'later','score':deepcopy(left['score'])}
        self.assertIs(runner.select_trial([left,right]),left)
        right['score']['mean_terminal_pose_loss']=np.nextafter(1.,0.)
        self.assertIs(runner.select_trial([left,right]),right)

    def test_all_ineligible_no_fallback(self):
        with self.assertRaises(ValueError):runner.select_trial([])
        with self.assertRaises(ValueError):runner.select_trial([{'score':{'eligible':False}}])

    def test_selection_does_not_mutate_trials(self):
        rows=[row('A'),row('B',0.,'planner_invalid')];before=deepcopy(rows)
        runner.trial_score(rows,['A','B']);self.assertEqual(rows,before)

    def test_planner_overrides_preserve_physical_model_and_cost_contract(self):
        c=load_pilot_config();before=deepcopy(c)
        settings={'geometry_proposal_enabled':True,'candidate_speed_norm_max_m_s':.03}
        p=runner.resolved_planner(c,{'candidate_count':128,'horizon':5,'block_lengths':[2,3]},settings)
        self.assertEqual(c,before)
        for key in ('physics','pusher','task','model','cost','geometry_baseline'):
            self.assertEqual(p[key],c[key])
        self.assertEqual(p['planner']['horizon'],5)
        p['planner']['block_lengths'][0]=999;self.assertEqual(c,before)

    def test_settings_fingerprint_rejects_changed_budget_or_final_test(self):
        settings=json.loads((ROOT/'configs/p3_planning_trials.v1.json').read_text())
        runner.validate_settings(settings)
        for key,value in (('screen_intervals',49),('final_test_used',True),('common_model_seeds',[11])):
            other=deepcopy(settings);other[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):runner.validate_settings(other)

    def test_compact_inputs_recreate_candidate_hash_without_detailed_arrays(self):
        c=load_pilot_config();c['planner'].update(candidate_count=16,horizon=5,block_lengths=[2,3])
        state=np.zeros(10,np.float64);state[6]=-.051
        task=TaskSpec.from_config(c);context=PlannerContext(np.array([.01,0.],np.float64),np.tile([.012,.002],(5,1)).astype(np.float64),7)
        def model(s,a):
            d=np.zeros_like(s);d[:,6:8]=a*.1;return d
        first=plan(state,task,context,keyed_planner_rng(999,8,7),model,c)
        identity={'state':state.tolist(),'rng_keys':[999,8,7,0],'planner_context_before':{
            'previous_applied_action':context.previous_applied_action.tolist(),'optimal_actions':context.optimal_actions.tolist(),'control_step':7}}
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'absent'
            compact=runner.retain_plan(folder,first,identity,full=False)
            self.assertFalse(folder.exists())
            prior=compact['planner_context_before']
            ctx=PlannerContext(np.asarray(prior['previous_applied_action'],np.float64),np.asarray(prior['optimal_actions'],np.float64),prior['control_step'])
            replay=plan(np.asarray(compact['state'],np.float64),task,ctx,keyed_planner_rng(*compact['rng_keys']),model,c)
            self.assertEqual(compact['candidate_hash'],runner.array_hash(replay.candidates.actions))
            self.assertEqual(compact['prediction_hash'],runner.array_hash(replay.predictions.states))
            self.assertEqual(compact['selected_actions'],replay.candidates.actions[replay.selected_index].tolist())

    def test_full_trace_retains_invalid_rows_and_failure_steps(self):
        c=load_pilot_config();c['planner'].update(candidate_count=16,horizon=5,block_lengths=[2,3])
        s=np.zeros(10,np.float64);s[6]=-.051
        def invalid(state,action):return np.full_like(state,np.nan)
        result=plan(s,TaskSpec.from_config(c),PlannerContext.initial(),keyed_planner_rng(1,2,3),invalid,c)
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'full';r=runner.retain_plan(folder,result,{},full=True)
            self.assertEqual(r['reason'],'planner_invalid');self.assertIsNone(r['selected_actions'])
            with np.load(folder/'trace.npz',allow_pickle=False) as saved:
                self.assertEqual(saved['candidates'].shape,(16,5,2))
                self.assertEqual(saved['valid'].sum(),0)
                self.assertTrue(np.isinf(saved['total_cost']).all())
                self.assertTrue((saved['failure_steps']==1).all())
                self.assertTrue(np.isnan(saved['raw_deltas']).all())

    def test_resource_denominator_and_action_dt_exceedances(self):
        s=runner.statistic([.001,.2,.05]);self.assertEqual(s['count'],3)
        self.assertEqual(s['over_action_dt_0_1s'],1);self.assertEqual(s['max_s'],.2)
        self.assertEqual(runner.statistic([])['count'],0)


if __name__=='__main__':unittest.main()
