def K_means(k_data, k_clusters = 2, ker = None):

    if (ker is None):
        raise Exception('Please specify what kernel you are using for k-means.')
    # Assign random labels
    if (k_clusters > len(k_data)):
        raise Exception(f'You expect more clusters than data?...' +
                        'Seems improbable. #DataElts = {len(k_data)} < #clusters = {k_clusters}')

    #We make sure that every cluster has at least one element
    current_labels = np.concatenate((np.linspace(0,k_clusters-1,k_clusters,dtype=int),
                                       np.random.randint(2, size=len(k_data)-k_clusters)))
    prev_labels = np.zeros(len(k_data),dtype=int)
    mu_norms = []
    cluster_vecs = [[] for i in range(k_clusters)]

    # Is this guaranteed to terminate?
    # Terminates when labels remain unchanged
    while(np.any(prev_labels != current_labels)):
        prev_labels = current_labels[:]

        # Group clusters for mean computations
        for i,v in enumerate(k_data):
             cluster_vecs[current_labels[i]].append(v)

        # d(\mu, \phi(x)) = ||\mu-\phi(x)||_{H}^2 = <\mu, \mu> + 2<\mu, \phi(x)> + <\phi(x),\phi(x)>
        # Hence, d(\mu, \phi(x)) = 1/n^2 \sum_{i,j} ker(x_i,x_j) + 2 \sum_{i} ker(x_i,x) + ker(x,x)

        # Firstly, compute ||\mu||_H^2 = \sum_{i,j} ker(x_i,x_j) for each cluster
        for i,c in enumerate(cluster_vecs):
            mu_norms.append(0)
            for c1, c2 in itertools.product(c,c):
                mu_norms[i] += ker(c1,c2)
            mu_norms[i] /= len(c)**2

        # Relabel according to distance to means
        for i,v in enumerate(k_data):
            # Compute <\mu, \phi(x)> and <\phi(x),\phi(x)> -- I am sorry for using x over v :(
            cur_label = cluster_vecs[current_lebels[i]]
            mu_dot_v = sum(list(map(lambda x: ker(x,v), cluster_vecs[cur_label])))/len(cluster_vecs[cur_label])
            v_dot_v = ker(v,v)

            min_dist = mu_norm[cur_label] + 2*mu_dot_v + v_dot_v
            #Inefficiency because we compute the distance to its current mean twice...
            for j, c in enumerate(cluster_vecs):
                # Vary over mu
                mu_dot_v = sum(list(map(lambda x: ker(x,v), cluster_vecs[j])))/len(cluster_vecs[j])

                cur_dist = mu_norm[j] + 2*mu_dot_v + v_dot_v
                if(cur_dist < min_dist):
                    current_labels[i] = j

    # Compute the errors
    errors = np.zeros(k_clusters)
    for i, c in enumerate(cluster_vecs):
        for v in c:
            mu_dot_v = sum(list(map(lambda x: ker(x,v), c)))/len(c)
            v_dot_v = ker(v,v)
            errors[i] += mu_norms[i] + 2*mu_dot_v + v_dot_v
            print(errors[i])

    return (current_labels,errors)