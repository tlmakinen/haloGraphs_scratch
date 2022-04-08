import cloudpickle as pickle
import jax
import jax.numpy as jnp
import numpy as onp
import jraph
import matplotlib.pyplot as plt

from utils import *

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

####################### GNN DATA ASSEMBLY FUNCTIONS #######################


def get_halo_batch(folder_name,
                  sim_index,
                  pad_nodes_to,
                  pad_edges_to,
                  mass_cut, # in units of Msun
                  connect_radius,
                  minspan=False,
                  minspan_neighborhood=False,
                  node_features=8,
                  edge_features=1,
                  savefile=None,
                  outfolder=None):

    n_sims = len(sim_index)
    mcut = mass_cut / 1e15
    #mass_cut = mcut*(10.**15)

    # padding for vmapping
    def get_padding(pad_nodes_to, pad_edges_to):
        nodes =  jnp.zeros((n_sims, pad_nodes_to, node_features))
        senders = jnp.zeros((n_sims, pad_edges_to), dtype=int)
        receivers = jnp.zeros((n_sims, pad_edges_to), dtype=int)
        edges = jnp.zeros((n_sims, pad_edges_to, edge_features))
        n_node = []
        n_edge = []
        _globals = None
        return nodes,senders,receivers,edges,n_node,n_edge,_globals


    nodes,senders,receivers,edges,n_node,n_edge,_globals = get_padding(pad_nodes_to,pad_edges_to)

    l = 0

    while l < len(sim_index):

        i = sim_index[l]

        mass,X,V,Npart = load_single_sim(folder_name, i, mass_cut)
        _nodes,_senders,_receivers,_edges, _nx, _n_edge = numpy_to_graph(X, V, mass,
                                                       Npart,return_components=True,
                                                       connect_radius=connect_radius,
                                                        minspan=minspan,
                                                        minspan_neighborhood=minspan_neighborhood)

        if _nx < pad_nodes_to:
            if _n_edge < pad_edges_to:
                nodes = nodes.at[l, :_nodes.shape[0], :].set(_nodes)
                senders = senders.at[l, :_senders.shape[0]].set(_senders)
                # this sets the filler indexes to a dummy node so we preserve computation on the actual graph values
                senders = senders.at[l, _senders.shape[0]:].set(jnp.squeeze(_nx).astype(int))

                receivers = receivers.at[l, :_receivers.shape[0]].set(_receivers)
                # this sets the filler indexes to a dummy node so we preserve computation on the actual graph values
                receivers = receivers.at[l, _receivers.shape[0]:].set(jnp.squeeze(_nx).astype(int))

                edges = edges.at[l, :_edges.shape[0], :].set(_edges)

                n_node.append(_nx) # these control how many edges / nodes get counted
                n_edge.append(_n_edge)
                l += 1

            else:
                print('boosting edge padding; \n restarting batch loop...')
                pad_edges_to += 10
                print('new edge padding length:', pad_edges_to)
                nodes,senders,receivers,edges,n_node,n_edge,_globals = get_padding(pad_nodes_to,pad_edges_to)
                l = 0

        else:
            print('boosting node padding; \n restarting batch loop...')
            pad_nodes_to += 10
            print('new node padding length:', pad_nodes_to)
            nodes,senders,receivers,edges,n_node,n_edge,_globals = get_padding(pad_nodes_to,pad_edges_to)
            l = 0




    n_node = jnp.array(n_node)
    n_edge = jnp.array(n_edge)



    # assemble explicitly batched GraphsTuple
    batched_graph = jraph.GraphsTuple(nodes=nodes, senders=senders, receivers=receivers,
                                edges=edges, n_node=n_node, n_edge=n_edge, globals=None)

    if savefile is not None:
        print('saving batched graph')
        savestem = 'rconnect_%d_mcut_%.2f'%(connect_radius, mcut)
        if outfolder is not None:
            savestem = outfolder + savestem
        save_obj(batched_graph, savestem + '_' + savefile)


    return batched_graph, (pad_nodes_to, pad_edges_to)


def get_training_data(connect_radius,
                      n_s=1000,
                      n_d=225,
                      mass_cut=1.5e15,
                      maindir='/data80/makinen/quijote/Halos/',
                      outfolder='/data80/makinen/quijote/Halos/imnn_data3/',
                      minspan=False,
                      minspan_neighborhood=False,
                      node_features=8,
                      edge_features=1,
                     ):
    np = onp
    sim_cov_index = np.arange(n_s)
    val_cov_index = np.arange(n_s, n_s*2)
    sim_derv_index = np.arange(n_d)
    val_derv_index = np.arange(start=n_d, stop=n_d*2)

    test_index = np.arange(start=450, stop=500)


    mass_cut = mass_cut # in units of Msun
    pad_nodes_to = 180 # could devise a function to pull in a dummy graph to get max nodes for padding # 500
    pad_edges_to = 400 # 2000
    node_features = 7


    # get fiducial
    savefile = 'fiducial'
    folder_name = maindir + savefile
    fiducial,_ = get_halo_batch(folder_name, sim_cov_index, mass_cut=mass_cut,
                                          connect_radius=connect_radius,
                                          minspan=minspan,
                                          minspan_neighborhood=minspan_neighborhood,
                                          node_features=node_features,
                                          pad_nodes_to=pad_nodes_to, pad_edges_to=pad_edges_to,
                                            savefile=savefile,
                                          outfolder=outfolder)

    # get VALIDATION fiducial
    savefile = 'fiducial'
    folder_name = maindir + savefile
    savefile = 'validation_' + savefile
    validation_fiducial,_ = get_halo_batch(folder_name, val_cov_index, mass_cut=mass_cut, connect_radius=connect_radius,
                                                 node_features=node_features,
                                                 pad_nodes_to=pad_nodes_to, pad_edges_to=pad_edges_to, savefile=savefile,
                                                 outfolder=outfolder)


    derivs = []
    # get regular derivatives
    print('getting training derivatives')
    for file in ['Om_m', 's8_m', 'Om_p', 's8_p']:
        folder_name = maindir + file

        _graph,_ = get_halo_batch(folder_name, sim_derv_index, mass_cut=mass_cut, connect_radius=connect_radius,
                                      node_features=node_features,
                                      pad_nodes_to=pad_nodes_to, pad_edges_to=pad_edges_to, savefile=None)
        derivs.append(_graph)


    numerical_derivative = jraph.batch(derivs)

    savestem = 'rconnect_%d_mcut_%.2f'%(connect_radius, mass_cut/1e15)
    savestem = outfolder + savestem
    save_obj(numerical_derivative, savestem + '_' + 'numerical_derivative')


    # get VALIDATION derivatives
    val_derivs = []
    print('getting validation derivatives')
    for file in ['Om_m', 's8_m', 'Om_p', 's8_p']:
        folder_name = maindir + file

        _graph,_ = get_halo_batch(folder_name, val_derv_index, mass_cut=mass_cut, connect_radius=connect_radius,
                                      node_features=node_features,
                                      pad_nodes_to=pad_nodes_to, pad_edges_to=pad_edges_to, savefile=None)
        val_derivs.append(_graph)


    validation_numerical_derivative = jraph.batch(val_derivs)

    savestem = 'rconnect_%d_mcut_%.2f'%(connect_radius, mass_cut/1e15)
    savestem = outfolder + savestem
    save_obj(numerical_derivative, savestem + '_' + 'validation_numerical_derivative')


    return fiducial,validation_fiducial,numerical_derivative,validation_numerical_derivative

def load_training_data(folder, connect_radius, mass_cut):
    fiducial = load_obj(folder + 'rconnect_%d_mcut_%.2f_%s.pkl'%(connect_radius, mass_cut/1e15,
                                                                 'fiducial'))
    validation_fiducial = load_obj(folder + 'rconnect_%d_mcut_%.2f_%s.pkl'%(connect_radius, mass_cut/1e15,
                                                                            'validation_fiducial'))
    numerical_derivative = load_obj(folder + 'rconnect_%d_mcut_%.2f_%s.pkl'%(connect_radius, mass_cut/1e15,
                                                                             'numerical_derivative'))
    validation_numerical_derivative = load_obj(folder + 'rconnect_%d_mcut_%.2f_%s.pkl'%(connect_radius, mass_cut/1e15,
                                                                              'validation_numerical_derivative'))
    return fiducial,validation_fiducial,numerical_derivative,validation_numerical_derivative
