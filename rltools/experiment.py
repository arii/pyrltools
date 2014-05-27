import numpy as np

def evaluate_trial(domain, agent):
    domain_copy = domain.copy()
    r, s_t = domain_copy.reset()
    cum_rew = r
    while s_t != None:
        r, s_t = domain_copy.step(agent.proposeAction(s_t))
        cum_rew += r

    return cum_rew

def evaluateAgent(domain, agent, num_trials):
    return [ evaluate_trial(domain, agent) for i in range(num_trials)]

def train_agent(domain, agent, num_steps, num_eval, eval_interval):
    score= []
    r, s_t = domain.reset()
    for i in xrange(num_steps):
        if i % eval_interval == 0:
            score.append( np.mean(evaluateAgent(domain, agent, num_eval)))

        if s_t == None:
            agent.reset()
            r, s_t = domain.reset()
        else:
            r, s_t = domain.step(agent.step(r, s_t))
    return score


def getRuns(**args):
    alpha = args.get('alpha')
    eta = args.get('eta')
    epsilon = args.get('epsilon')
    gamma = args.get('gamma')
    mommentum = args.get('mommentum')
    num_runs = args.get('num_runs', 10)

    domain_factory = args.get('domain_factory')
    projector_factory = args.get('project_factory')
    policy_factory = args.get('policy_factory')
    value_fn_factory = args.get('valuefn_factory')
    agent_factory = args.get('agent_factory')

    param = {'alpha':alpha,
             'eta':eta,
             'epsilon':epsilon,
             'gamma':gamma,
             'mommentum':mommentum}

    for i in range(num_runs):
        domain = domain_factory(**args)
        projector = projector_factory(domain = domain, **args)
        param['layers'] = [projector.size] + args.get('layers', [40]) + [1]
        valuefn = value_fn_factory(projector = projector, **param)
        policy = policy_factory(domain = domain, valuefn = valuefn, **args)
        agent = agent_factory(policy = policy, valuefn = valuefn, **args)

        yield domain, agent

def get_score_list(**args):
    num_train_steps = args.get('num_train_steps')
    num_eval_trial = args.get('num_eval_trial')
    eval_interval = args.get('eval_interval')
    return map(lambda (d,a): train_agent(d, a, num_train_steps,
                                                    num_eval_trial,
                                                    eval_interval),
                            getRuns(**args))
