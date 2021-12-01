import dace
from dace.frontend.common import op_repository as oprepo

@oprepo.replaces('warpReduce_sum')
def warpReduce_sum(pv, sdfg: dace.SDFG, state: dace.SDFGState, x: str) -> str:
   desc = sdfg.arrays[x]
   newname, _ = sdfg.add_temp_transient(desc.shape, desc.dtype, desc.storage)
   ctype = desc.dtype.ctype

   t = state.add_tasklet(
       'warpReduce', {'__a'}, {'__out'}, f'''
       __out = dace::warpReduce<dace::ReductionType::Sum, {ctype}>::reduce(__a);
   ''', dace.Language.CPP)
   r = state.add_read(x)
   w = state.add_write(newname)
   state.add_edge(r, None, t, '__a', dace.Memlet(data=x))
   state.add_edge(t, '__out', w, None, dace.Memlet(data=newname))
   return newname