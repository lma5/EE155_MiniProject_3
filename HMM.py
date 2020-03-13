########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang
# Description:  Set 5 solutions
########################################

import random

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.

            D:          Number of observations.

            A:          The transition matrix.

            O:          The observation matrix.

            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    Output sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

   
        for i in range(self.L):
            probs[1][i] = self.O[i][x[0]] * self.A_start[i]
            seqs[1][i] = str(i)
        
        
        for length in range(2,M+1):
            for col in range(self.L):
                # c is the previous ending state
                # col-column indicates the current ending state
                # all possibilities: for any previous state, *transaction prob *emitting prob given the current state
                possi = [probs[length-1][c] * self.A[c][col] * self.O[col][x[length-1]] for c in range(self.L)]
                probs[length][col] = max(possi)
                seqs[length][col] = seqs[length-1][possi.index(max(possi))]+str(col)
                    
        max_col_idx = probs[-1].index(max(probs[-1]))
        max_seq =  seqs[-1][max_col_idx]
        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        
        # c here is the initial state
        alphas[1] = [self.O[c][x[0]] * self.A_start[c] for c in range(self.L)]
        for length in range(2,M+1):
            for col in range(self.L):
                # c is the previous ending state
                # col-column indicates the current ending state
                # αz (i+1) = Oxi+1,z∑α j (i)Az, j
                alphas[length][col] = sum([alphas[length-1][c] * self.A[c][col] * self.O[col][x[length-1]] for c in range(self.L)])
                
            if normalize:
                alphas[length] = [alpha / sum(alphas[length]) for alpha in alphas[length]]
        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        
        betas[M] = [1 for _ in range(self.L)]
        for length in range(M-1,0,-1):
            for col in range(self.L):
                # c is the last ending state
                # col-column indicates the current ending state
                betas[length][col] = sum([betas[length+1][c] * self.A[col][c] * self.O[c][x[length]] for c in range(self.L)])
            if normalize:
                betas[length] = [beta / sum(betas[length]) for beta in betas[length]]
       
        return betas


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        # Note that a comment starting with 'E' refers to the fact that
        # the code under the comment is part of the E-step.

        # Similarly, a comment starting with 'M' refers to the fact that
        # the code under the comment is part of the M-step.

        # M training examples
        M = len(X)
        # each example is X_i
        P_ijab = [[[[0. for _ in range(self.L)] for _ in range(self.L)]
                    for _ in range(len(seq)-1)] for seq in X]
        P_ija = [[[0. for _ in range(self.L)] for _ in range(len(seq))] for seq in X]
        
        for iteration in range(N_iters):
            print(iteration)
            #=======================================
            # Expectation step
            for i in range(M):  
                alphas = self.forward(X[i], True)
                betas = self.backward(X[i], True)
                # each stage of an X example
                for j in range(1, len(X[i])+1):
                    # slide P75
                    # For each training x=(x^1,...,x^M)
                    # Computes each P(y^i=z|x) for y=(y^1,...,y^M)
                    # for each y^i=z, z=a (self.L possible states)
                    #initialize 0 for numerator alpha_z(j)*beta_z(j) for each value of z(a)
                    numer = [0 for _ in range(self.L)]
                    for a in range(self.L):
                        numer[a] = alphas[j][a]*betas[j][a] 
                    # the denominator is the sum of alpha_z'(j)*beta_z'(j) for every z'
                    denom = sum(numer)
                    # the demoninator is ths same for every a since it's the sum
                    P_ija[i][j-1] = [x/denom for x in numer]
                    
                    
                    if j == len(X[i]):
                        break
                    
                    #slide P72
                    # y^i=a, y^(i+1)=b
                    numer2 = [[0 for _ in range(self.L)] for _ in range(self.L)]
                    # P(y^(i+1) =b|y^(i) =a) = self.A[a][b] from state a to state b
                    # P(x^i |y^i =b) = Prob(emitting seq element x given state y) = self.O[b][seq element]
                    for a in range(self.L):
                        for b in range(self.L):
                            numer2[a][b] = alphas[j][a]*betas[j+1][b]*self.A[a][b]*self.O[b][X[i][j]]
                    # the denominator2 is the sum of 2D-array numerator2 for every a',b'
                    denom2 = sum([sum(row) for row in numer2])
                    # the demoninator2 is ths same for every a since it's the sum
                    P_ijab[i][j-1] = [[x/denom2 for x in numer2_row] for numer2_row in numer2]
                    
            #=======================================
            # Maximization step
            # slide p69
            # update self.A
            for a in range(self.L):
                for b in range(self.L):
                    a_cnt = 0
                    b_cnt = 0
                    for i in range(M):
                        for j in range(len(X[i])-1):
                            a_cnt+=P_ijab[i][j][a][b]
                            b_cnt+=P_ija[i][j][a]
                            
                    self.A[a][b] = a_cnt/b_cnt if b_cnt else 0
                    
            # update self.O
            for z in range(self.L):
                for w in range(self.D):
                    w_cnt = 0
                    z_cnt = 0
                    for i in range(M):
                        for j in range(len(X[i])):
                            z_cnt+=P_ija[i][j][z]
                            if X[i][j]==w:
                                w_cnt += P_ija[i][j][z]
                    
                    self.O[z][w] = w_cnt/z_cnt if z_cnt else 0

    
    def generate_emission(self, M, obs_map_r, syl_dict):
        '''
        Generates an emission of M syllables, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            M:          Number of Syllables

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        state = random.choice(range(self.L))
        states = []
        # counter keeping track of number of syllables
        count = 0

        while count < M:
            # Append state.

            if count == 0:
                states.append(state)
            else:
                # Sample next state.
                rand_var = random.uniform(0, 1)
                next_state = 0

                while rand_var > 0:
                    rand_var -= self.A[state][next_state]
                    next_state += 1

                next_state -= 1
                state = next_state
                states.append(state)

            while True:
                # Sample next observation.
                rand_var = random.uniform(0, 1)
                next_obs = 0

                while rand_var > 0:
                    rand_var -= self.O[state][next_obs]
                    next_obs += 1

                next_obs -= 1
                # amount of syllables needed
                diff = M - count
                word = obs_map_r[next_obs]
                syllable = self.find_syl(word, syl_dict, diff)
                if syllable != -1:
                    count += syllable
                    break

            emission.append(next_obs)

        return emission, states
    
        
    def generate_emission_with_start(self, M, obs_map, obs_map_r, syl_dict, start_word, start_state):
        '''
        Generates an emission of M syllables, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            M:          Number of Syllables

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        state = start_state
        states = []
        # counter keeping track of number of syllables
        count = 0

        while count < M:
            # Append state.

            if count == 0:
                states.append(state)
                emission.append(obs_map[start_word])
                syllable = self.find_syl(start_word, syl_dict, M)
                count += syllable

            else:
                # Sample next state.
                rand_var = random.uniform(0, 1)
                next_state = 0

                while rand_var > 0:
                    rand_var -= self.A[state][next_state]
                    next_state += 1

                next_state -= 1
                state = next_state
                states.append(state)

            while True:
                # Sample next observation.
                rand_var = random.uniform(0, 1)
                next_obs = 0

                while rand_var > 0:
                    rand_var -= self.O[state][next_obs]
                    next_obs += 1

                next_obs -= 1
                # amount of syllables needed
                diff = M - count
                word = obs_map_r[next_obs]
                syllable = self.find_syl(word, syl_dict, diff)
                if syllable != -1:
                    count += syllable
                    break

            emission.append(next_obs)

        return emission, states


    

    def find_state(self, x):
        '''
        Chooses a state (based on observation matrix) that could have
        generated a given emission x.
        '''

        # Find P(y | x) and sample state from the probs
        probs = [row[x] for row in self.O]
        sum_probs = sum(probs)
        probs = [prob / sum_probs for prob in probs]
        rand_var = random.uniform(0, 1)
        next_state = 0
        while rand_var > 0:
            rand_var -= probs[next_state]
            next_state += 1

        return next_state - 1


    def find_syl (self, word, dic, diff):
        '''
        Returns number of syllables of a word. Returns -1
        if the number of syllables in the word makes
        the line exceed 10 syllables
        '''

        lst = dic[word.lower()]

        real = lst[0]
        end = lst[1]


        # threshold = 0
        # bool found = False
        for i in range(len(real)):
            if real[i] <= diff:
                return random.choice(real[i:])

        if len(end) != 0:
            for j in range(len(end)):
                if end[j] == diff:
                    return end[j]

        return -1


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the output sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any output sequence, i.e. the
        # probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(0) gives the probability of the output sequence. Summing
        # this over all states and then normalizing gives the total
        # probability of x paired with any output sequence, i.e. the
        # probability of x.
        prob = sum([betas[1][k] * self.A_start[k] * self.O[k][x[0]] \
            for k in range(self.L)])

        return prob


def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.

        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrices A and O.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
