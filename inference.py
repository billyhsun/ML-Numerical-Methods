import numpy as np
import graphics
import rover


# Forward-backward algorithm implementation
def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    # TODO: Compute the forward messages
    #forward messages are a distribution
    #update distribution based on observation and transition model
    #print(observations)
    '''copy = prior_distribution #copy is alpha distribution
    print(copy)
    print(observations)
    for state in copy:
        observed_given_hidden = observation_model(state)
        for obsstate in observed_given_hidden:
            copy[state] = copy[state]*observed_given_hidden[obsstate]'''

    alpha_z0 = prior_distribution
    for state in prior_distribution:
        likelihood_function = observation_model(state) #p(xi, yi| z0)
        p_obs_given_hidden = likelihood_function[observations[0]] #p(first observation | z0)
        alpha_z0[state] = prior_distribution[state]*p_obs_given_hidden #p(first observation|z0) * p(z0) proportional to p(z0|x0,y0)
    alpha_z0.renormalize()
    forward_messages[0] = alpha_z0

    for i in range(1,num_time_steps):

        alpha_zi_1 = forward_messages[i-1] #just want the shapes to match previous timestep distribution
        xi_yi = observations[i]
        alpha_zi = rover.Distribution()
        #print(xi_yi)

        for zi in all_possible_hidden_states:
            sum = 0
            likelihood_function = observation_model(zi) #p(xn,yn|zi)
            if xi_yi is not None:
                p_obs_given_hidden = likelihood_function[xi_yi] #p(xi,yi|zi) were xi,yi is our observation
            else:
                p_obs_given_hidden = 1
            for zi_1 in alpha_zi_1:
                nextgivenprev = transition_model(zi_1)
                sum += alpha_zi_1[zi_1]*nextgivenprev[zi]
            if p_obs_given_hidden*sum>0.0:
                alpha_zi[zi] = p_obs_given_hidden * sum

        #print(alpha_zi)
        alpha_zi.renormalize()

        sum = 0
        for key in alpha_zi:
            sum += alpha_zi[key]
        #print(sum)
        forward_messages[i] = alpha_zi


    # TODO: Compute the backward messages

    #Initialize the last beta

    beta_zN_1 = prior_distribution
    for state in all_possible_hidden_states:
        beta_zN_1[state] = 1
    beta_zN_1.renormalize()
    backward_messages[-1] =  beta_zN_1
    
    #Recursively go back for other betas
    beta_zn_1 = beta_zN_1

    print("PRINTING BETA VALUES NOW")
    for n in range(num_time_steps-2, -1, -1):
        
        beta_zn = backward_messages[n+1]
        beta_zn_1 = rover.Distribution()
        xn_yn = observations[n + 1]
        #print(xn_yn)

        for zn_1 in all_possible_hidden_states:
            sum = 0
            nextgivenprev = transition_model(zn_1)
            for zn in beta_zn:

                likelihood_function = observation_model(zn)
                if xn_yn is not None:
                    p_obs_given_hidden = likelihood_function[xn_yn]
                else:
                    p_obs_given_hidden = 1
                sum += beta_zn[zn]*p_obs_given_hidden*nextgivenprev[zn]
            if sum > 0.0:
                beta_zn_1[zn_1] = sum
        #print(beta_zn_1)
        beta_zn_1.renormalize()
        backward_messages[n] = beta_zn_1

    
    # TODO: Compute the marginals 
    print("PRINTING MARGINALS")
    for i in range(num_time_steps):
        print(i)
        alpha_zi = forward_messages[i]
        beta_zi = backward_messages[i]
        gamma_zi = rover.Distribution()
        for state in all_possible_hidden_states:
            if alpha_zi[state]*beta_zi[state] > 0:
                gamma_zi[state] = alpha_zi[state]*beta_zi[state]
        gamma_zi.renormalize()


        marginals[i] = gamma_zi

        print(marginals[i])
        #print(marginals[i].get_mode())
    return marginals


# Helper functions for Viterbi
def safe_log(x):
    if x == 0:
        return -np.inf
    else:
        return np.log(x)


def get_max(Zn_1, W_prev):
    max_log_prob = -np.inf
    arg_max = None
    for prev_state in W_prev:
        curr = safe_log(rover.transition_model(prev_state)[Zn_1]) + W_prev[prev_state]
        if curr > max_log_prob:
            max_log_prob = curr
            arg_max = prev_state
    return max_log_prob, arg_max


def get_mode(x):
    maximum = -np.inf
    arg_max = None
    for key in x.keys():
        if x[key] > maximum:
            arg_max = key
            maximum = x[key]
    return arg_max, maximum


# Viterbi algorithm implementation
def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    # Stores all Wi's
    W = [None] * len(observations)
    map_sequence = []

    # Key is Z_n+1 and value is Z_n hidden state
    links = []
    for t in range(len(observations)):
        newdict = dict()
        links.append(dict())

    # Calculate W0
    p_z0 = prior_distribution
    W0 = rover.Distribution()
    log_p_xz = rover.Distribution()
    for hidden_state in p_z0:
        likelihood = observation_model(hidden_state)
        if (observations[0] is None):
            log_p_xz[hidden_state] = 0
        else:
            log_p_xz[hidden_state] = safe_log(likelihood[observations[0]])
        W0[hidden_state] = log_p_xz[hidden_state] + safe_log(p_z0[hidden_state])
        #print(W0[hidden_state])

    #print(len(W0))
    W[0] = W0

    # Calculate Wi
    for t in range(1, len(observations)):
        W_t = rover.Distribution()
        log_p_xz = rover.Distribution()
        for hidden_state in all_possible_hidden_states:
            likelihood = observation_model(hidden_state)
            if (observations[t] is None):
                log_p_xz[hidden_state] = 0
            else:
                log_p_xz[hidden_state] = safe_log(likelihood[observations[t]])
            max_log_prob, arg_max = get_max(hidden_state, W[t - 1])
            if arg_max != None:
                '''print(t)
                print(hidden_state)
                print(links[t])

                while True:
                    pass'''



                links[t][hidden_state] = arg_max
                W_t[hidden_state] = log_p_xz[hidden_state] + max_log_prob
            # print(W_t[hidden_state])
        arg_max_debug, max_value = get_mode(W_t)
        print("t: {} | arg: {} | max: {}".format(t, arg_max_debug, max_value))
        W[t] = W_t

    final_dist = W[-1]
    max_hidden_state, max_val = get_mode(final_dist)
    map_sequence.append(max_hidden_state)
    for t in range(len(observations) - 1, 0, -1):
        prev = map_sequence[-1]
        print("T =", t)
        print("prev", prev)
        #print(prev)
        curr = links[t][prev]
        print("curr", curr)
        print("\n")
        #print(links[t])
        map_sequence.append(curr)
        # print(curr)
    estimated_hidden_states = map_sequence[::-1]

    for d in range (1, len(links)):
        if links[d] == links[d-1]:
            print("FatalError", d)

    return estimated_hidden_states


if __name__ == '__main__':
   
    enable_graphics = True
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')


   
    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])
  
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()

