"""CPU C++ vs NumPy/ctypes; optional only when no C++ compiler is available."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from poppim.encoding.bitpacking import encode_query,encode_keys
from poppim.scoring.reference import score_encoded


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('g++ unavailable: scalar C++ cross-check not run')
    output = tmp_path_factory.mktemp('oracle')/'oracle.so'
    subprocess.run([compiler,'-std=c++17','-O2','-ffp-contract=off','-shared','-fPIC',
                    str(Path(__file__).parent/'oracles/score_oracle.cpp'),'-o',str(output)],check=True)
    lib = ctypes.CDLL(str(output))
    lib.scalar_scores.argtypes = [ctypes.c_void_p]*8 + [ctypes.c_int]*3 + [ctypes.c_void_p]
    lib.scalar_scores.restype = None
    return lib.scalar_scores


@pytest.mark.parametrize('group',[16,32,64])
@pytest.mark.parametrize('dim',[33,127,128])
def test_cpp_bitcounts_and_fma_sequence(oracle,group,dim):
    rng = np.random.default_rng(778)
    q = encode_query(rng.normal(size=(1,1,dim)).astype(np.float16),group)
    k = encode_keys(rng.normal(size=(1,67,dim)).astype(np.float16),group,32)
    arrays = [np.ascontiguousarray(a,dtype=dtype) for a,dtype in
              [(q.sign,np.uint32),(q.magnitude,np.uint32),(q.low,np.float32),
               (q.delta,np.float32),(k.sign,np.uint32),(k.magnitude,np.uint32),
               (k.low,np.float32),(k.delta,np.float32)]]
    output = np.empty(67,dtype=np.float32)
    oracle(*(a.ctypes.data for a in arrays),dim,group,67,output.ctypes.data)
    expected = output.astype(np.float16) if dim==128 else output
    np.testing.assert_array_equal(score_encoded(q,k)[0,0],expected)
