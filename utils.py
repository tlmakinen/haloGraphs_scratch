from functools import partial
import jax
import jax.numpy as jnp
import jraph
from scipy.sparse import csgraph
import numpy as np
import os, sys
from struct import unpack

import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


####################### GRAPH ASSEMBLY FUNCTIONS #######################

def get_distances(X):
    nx = X.shape[0]
    return (X[:, None, :] - X[None, :, :])[jnp.tril_indices(nx, k=-1)]


def get_receivers_senders(nx, dists, connect_radius=100):
    '''connect nodes within `connect_radius` units'''
    
    senders,receivers = jnp.tril_indices(nx, k=-1)
    mask = dists[jnp.tril_indices(nx, k=-1)] < connect_radius
    return senders[mask], receivers[mask], dists

def get_minspan_receivers_senders(nx, dists, connect_radius=None):
    """connect graph via minimum spanning tree"""
    min_span_tree = csgraph.minimum_spanning_tree(dists, overwrite=False)
    dists = min_span_tree.toarray()
    senders,receivers = np.where(dists>0.)
    
    return senders, receivers, dists

def get_minspan_neighborhoods(nx, dists, connect_radius=100):
    
    # first get minumum spanning tree
    min_span_tree = csgraph.minimum_spanning_tree(dists, overwrite=False)
    min_span_tree = min_span_tree.toarray() 

    
    # then get neighborhood connections
    senders,receivers = jnp.tril_indices(nx, k=-1)
    mask = dists > connect_radius
    dists = dists.at[mask].set(0.) # set all large distances to zero
    
    # add the spanning tree
    dists = dists.at[mask].set(min_span_tree[mask])
    
    senders,receivers = np.where(dists>0.)
    
    return senders, receivers, dists


def get_r2(X):
    """calculate euclidean distance from positional information"""
    nx = X.shape[0]
    alldists = jnp.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    return alldists #[jnp.tril_indices(nx, k=-1)]


def numpy_to_graph(X, V, masses, 
                   Npart,
                   connect_radius=50, 
                   minspan=False,
                   minspan_neighborhood=True,
                   return_components=True,
                   scale_inputs=True, 
                   nx=None):
    """Assemble halo graph attributes

    Parameters
    ----------
    X : np.array
        positional information in shape (None, 3)
        
    V : np.array
        velocity information in shape (None, 3)
    
    connect_radius : float
        Radius (Mpc/h) within which to connect neighboring halos
        
    minspan : bool
        Whether to connect the graph via a minimum spanning tree
        
    minspan_neighborhood : bool
        If True, connects graph first with a minumum spanning tree 
        such that all nodes are connected. Then connects all nodes within a
        neighborhood specified by `connect_radius`
        
    return_components : bool
        Whether to return graph components as numpy arrays (default)
        for explicit batching or as single graphs
    
    scale_inputs : bool
        Whether to scale graph attributes to neural network-friendly 
        dynamical ranges
        
    nx : int
        Number of input nodes (deprecated)
        
    Returns
    -------
    if return_components is True:
        tuple:
            graph attributes: (nodes,senders,receivers,edges, n_node, n_edge)
    
    else:
        jraph.GraphsTuple:
            assembled jraph graph with attributes

    """
    
    
    if nx is None:
        nx = jnp.array([X.shape[0]])
        
    _nx = jnp.array([nx])
    
    masses = jnp.array(masses)
    Npart = jnp.array(Npart)
    
    if scale_inputs:
        X /= 1000. # in units of Gpc
        connect_radius /= 1000. # in units of Gpc
        V /= 1000. 
        #masses = jnp.log(masses)
        Npart = jnp.log(Npart)
        Npart = (Npart - jnp.mean(Npart)) / jnp.std(Npart)
    
    # CHOOSE EDGE CALCULATOR
    
    if minspan:
        get_rs = get_minspan_receivers_senders   
    elif minspan_neighborhood:
        get_rs = get_minspan_neighborhoods
    else:
        get_rs = get_receivers_senders
    
    # mask out halos with distances < connect_radius
    dists = get_r2(X)
    
    receivers, senders, dists = get_rs(nx, dists, connect_radius=connect_radius)
    
    edges = dists #[dists < connect_radius]

    receivers = jnp.array(receivers)
    senders = jnp.array(senders)

    if masses is None:
        # Default all masses to one
        masses = jnp.ones(nx)
    elif isinstance(masses, (int, float)):
        masses = masses*jnp.ones(nx)
    else:
        assert len(masses) == nx, 'Wrong size for masses'


    nodes = jnp.concatenate([masses.reshape([-1, 1]), X, V], axis=1)
    
    if return_components:
        return nodes,senders,receivers,edges[:, None], nx, jnp.array(edges.shape[0])
    
    else:
        graph = jraph.GraphsTuple(nodes=nodes, senders=senders, receivers=receivers,
                                  edges=edges[:, None], n_node=_nx, n_edge=jnp.array([edges.shape[0]]), globals=None)
        return graph
    
    
    
def load_single_sim(folder_name, 
             sim_num,
             mass_cut=2e15,
             snapnum=4):
    # input files
    snapdir = folder_name + '/%d/'%(sim_num) #folder hosting the catalogue
    snapnum = snapnum    #vredshift

    # determine the redshift of the catalogue
    z_dict = {4:0.0, 3:0.5, 2:1.0, 1:2.0, 0:3.0}
    redshift = z_dict[snapnum]

    # read the halo catalogue
    FoF = FoF_catalog(snapdir, snapnum, long_ids=False,
                              swap=False, SFR=False, read_IDs=False)

    # get the properties of the halos
    pos_h = FoF.GroupPos/1e3            # Halo positions in Mpc/h
    mass  = FoF.GroupMass*1e10          # Halo masses in Msun/h
    vel_h = FoF.GroupVel*(1.0+redshift) # Halo peculiar velocities in km/s
    Npart = FoF.GroupLen                # Number of CDM particles in the halo
    
    mass_cut = (mass > mass_cut)
    
    return mass[mass_cut],pos_h[mass_cut],vel_h[mass_cut],Npart[mass_cut]



def update_edge_dummy(edge, sender_node, receiver_node, globals_):
    return edge


def update_node_dummy(node, sender, receiver, globals_):
    return node

####################### VISUALIZE GRAPH #######################
    
def plot_graph(graph,
                ax=None):
    """visualize jraph graph using Networkx"""
    
    send_receive = [(int(graph.senders[l]), int(graph.receivers[l])) for l in range(len(graph.receivers))]
    
    G = nx.Graph()
    G.add_nodes_from(list(np.arange(graph.nodes[:, :1].shape[0])))
    G.add_edges_from(send_receive)


    # 3d spring layout
    pos = graph.nodes[:, 1:4] #X 
    
    masses = graph.nodes[:, 0]
    
    
    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    # Create the 3D figure
    fig = plt.figure(figsize=(7,4))
    
    if ax is None:
        ax = fig.add_subplot(111, projection="3d")

    # Plot the nodes - alpha is scaled by "depth" automatically
    sc = ax.scatter(*node_xyz.T, s=45, ec='w', c=masses, cmap='gist_gray')



    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray", lw=1.5)


    def _format_axes(ax):
        """Visualization options for the 3D axes."""
        # Turn gridlines off
        ax.grid(False)
        # Suppress tick labels
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
        # Set axes labels
        ax.set_xlabel(r"$x$", fontsize=15)
        ax.set_ylabel(r"$y$", fontsize=15)
        ax.set_zlabel(r"$z$", fontsize=15)


    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')


    ax.view_init(azim=15, elev=20)
    _format_axes(ax)  
    plt.show()
    
    return ax


####################### GRAPH DATASET ASSEMBLY #######################



########################### HALO READ SCRIPT #############################

class FoF_catalog:
    """Pylians FoF catalog read script
    see https://pylians3.readthedocs.io/en/master/index.html
    for more information"""
    def __init__(self, basedir, snapnum, long_ids=False, swap=False,
                 SFR=False, read_IDs=True, prefix='/groups_'):

        if long_ids:  format = np.uint64
        else:         format = np.uint32

        exts=('000'+str(snapnum))[-3:]

        #################  READ TAB FILES ################# 
        fnb, skip, Final = 0, 0, False
        dt1 = np.dtype((np.float32,3))
        dt2 = np.dtype((np.float32,6))
        prefix = basedir + prefix + exts + "/group_tab_" + exts + "."
        while not(Final):
            f=open(prefix+str(fnb), 'rb')
            self.Ngroups    = np.fromfile(f, dtype=np.int32,  count=1)[0]
            self.TotNgroups = np.fromfile(f, dtype=np.int32,  count=1)[0]
            self.Nids       = np.fromfile(f, dtype=np.int32,  count=1)[0]
            self.TotNids    = np.fromfile(f, dtype=np.uint64, count=1)[0]
            self.Nfiles     = np.fromfile(f, dtype=np.uint32, count=1)[0]

            TNG, NG = self.TotNgroups, self.Ngroups
            if fnb == 0:
                self.GroupLen    = np.empty(TNG, dtype=np.int32)
                self.GroupOffset = np.empty(TNG, dtype=np.int32)
                self.GroupMass   = np.empty(TNG, dtype=np.float32)
                self.GroupPos    = np.empty(TNG, dtype=dt1)
                self.GroupVel    = np.empty(TNG, dtype=dt1)
                self.GroupTLen   = np.empty(TNG, dtype=dt2)
                self.GroupTMass  = np.empty(TNG, dtype=dt2)
                if SFR:  self.GroupSFR = np.empty(TNG, dtype=np.float32)
                    
            if NG>0:
                locs=slice(skip,skip+NG)
                self.GroupLen[locs]    = np.fromfile(f,dtype=np.int32,count=NG)
                self.GroupOffset[locs] = np.fromfile(f,dtype=np.int32,count=NG)
                self.GroupMass[locs]   = np.fromfile(f,dtype=np.float32,count=NG)
                self.GroupPos[locs]    = np.fromfile(f,dtype=dt1,count=NG)
                self.GroupVel[locs]    = np.fromfile(f,dtype=dt1,count=NG)
                self.GroupTLen[locs]   = np.fromfile(f,dtype=dt2,count=NG)
                self.GroupTMass[locs]  = np.fromfile(f,dtype=dt2,count=NG)
                if SFR:
                    self.GroupSFR[locs]=np.fromfile(f,dtype=np.float32,count=NG)
                skip+=NG

                if swap:
                    self.GroupLen.byteswap(True)
                    self.GroupOffset.byteswap(True)
                    self.GroupMass.byteswap(True)
                    self.GroupPos.byteswap(True)
                    self.GroupVel.byteswap(True)
                    self.GroupTLen.byteswap(True)
                    self.GroupTMass.byteswap(True)
                    if SFR:  self.GroupSFR.byteswap(True)
                        
            curpos = f.tell()
            f.seek(0,os.SEEK_END)
            if curpos != f.tell():
                raise Exception("Warning: finished reading before EOF for tab file",fnb)
            f.close()
            fnb+=1
            if fnb==self.Nfiles: Final=True


        #################  READ IDS FILES ################# 
        if read_IDs:

            fnb,skip=0,0
            Final=False
            while not(Final):
                fname=basedir+"/groups_" + exts +"/group_ids_"+exts +"."+str(fnb)
                f=open(fname,'rb')
                Ngroups     = np.fromfile(f,dtype=np.uint32,count=1)[0]
                TotNgroups  = np.fromfile(f,dtype=np.uint32,count=1)[0]
                Nids        = np.fromfile(f,dtype=np.uint32,count=1)[0]
                TotNids     = np.fromfile(f,dtype=np.uint64,count=1)[0]
                Nfiles      = np.fromfile(f,dtype=np.uint32,count=1)[0]
                Send_offset = np.fromfile(f,dtype=np.uint32,count=1)[0]
                if fnb==0:
                    self.GroupIDs=np.zeros(dtype=format,shape=TotNids)
                if Ngroups>0:
                    if long_ids:
                        IDs=np.fromfile(f,dtype=np.uint64,count=Nids)
                    else:
                        IDs=np.fromfile(f,dtype=np.uint32,count=Nids)
                    if swap:
                        IDs=IDs.byteswap(True)
                    self.GroupIDs[skip:skip+Nids]=IDs[:]
                    skip+=Nids
                curpos = f.tell()
                f.seek(0,os.SEEK_END)
                if curpos != f.tell():
                    raise Exception("Warning: finished reading before EOF for IDs file",fnb)
                f.close()
                fnb+=1
                if fnb==Nfiles: Final=True


# This function is used to write one single file for the FoF instead of having
# many files. This will make faster the reading of the FoF file
def writeFoFCatalog(fc, tabFile, idsFile=None):
    if fc.TotNids > (1<<32)-1: raise Exception('TotNids overflow')

    f = open(tabFile, 'wb')
    np.asarray(fc.TotNgroups).tofile(f)
    np.asarray(fc.TotNgroups).tofile(f)
    np.asarray(fc.TotNids, dtype=np.int32).tofile(f)
    np.asarray(fc.TotNids).tofile(f)
    np.asarray(1, dtype=np.uint32).tofile(f)
    fc.GroupLen.tofile(f)
    fc.GroupOffset.tofile(f)
    fc.GroupMass.tofile(f)
    fc.GroupPos.tofile(f)
    fc.GroupVel.tofile(f)
    fc.GroupTLen.tofile(f)
    fc.GroupTMass.tofile(f)
    if hasattr(fc, 'GroupSFR'):
        fc.GroupSFR.tofile(f)
    f.close()

    if idsFile:
        f = open(idsFile, 'wb')
        np.asarray(fc.TotNgroups).tofile(f)
        np.asarray(fc.TotNgroups).tofile(f)
        np.asarray(fc.TotNids, dtype=np.uint32).tofile(f) 
        np.asarray(fc.TotNids).tofile(f)
        np.asarray(1, dtype=np.uint32).tofile(f)
        np.asarray(0, dtype=np.uint32).tofile(f) 
        fc.GroupIDs.tofile(f)
        f.close()