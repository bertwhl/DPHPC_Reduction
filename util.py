import dace
from dace.frontend.common import op_repository as oprepo

def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
   """ Finds the first map entry node by the given parameter name. """
   return next(n for n, _ in sdfg.all_nodes_recursive()
               if isinstance(n, dace.nodes.MapEntry) and pname in n.params)

@oprepo.replaces('sync_threads')
def sync_threads(pv, sdfg: dace.SDFG, state: dace.SDFGState):
    state.add_tasklet(name='syncronize_threads',
                      inputs={},
                      outputs={},
                      code='__syncthreads();', 
                      language=dace.Language.CPP)