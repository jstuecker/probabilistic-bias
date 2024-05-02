import sympy
import numpy as np

def swapax(M, ax1, ax2):
    """Swaps two axes or two lists of axes"""
    ax1, ax2, = np.atleast_1d(ax1, ax2)

    res = np.copy(M)
    for i in range(0,len(ax1)):
        res = np.swapaxes(res, ax1[i], ax2[i])
    return res

def symmetrize(M, axs):
    """Symmetrizes a Matrix so that it is invariant under
    swapping any of the indicated axes
    e.g. 
    symmetrize(M, (0,1,2))
    """
    if(len(axs) <= 1):
        return M
    MsymNminus1 = symmetrize(M, axs[1:])
    
    Mnew = np.copy(MsymNminus1)
    for i in range(1, len(axs)):
        Mnew += swapax(MsymNminus1, axs[0], axs[i])
    return Mnew / len(axs)

def symmetrizeGroups(M, axsgroups, sym_equal_len_groups=True):
    """Symmetrizes several groups of a Matrix. Optionally also
    symmetrizes between permuting different groups that have 
    equal lengths
    
    e.g. 
    symmetrize(M, ((0,1), (2,3), (4,5)))
    """
    
    Ms = M
    for axs in axsgroups:
        Ms = symmetrize(Ms, axs)
    
    if sym_equal_len_groups:
        # additionally symmetrize all groups with the same length
        lens = np.array([len(axs) for axs in axsgroups])
        
        maxlen = np.max(lens)
        for nax in range(2, maxlen+1):
            isel = np.where(lens==nax)[0]
            axsgroups_with_nax = [axsgroups[i] for i in isel]
            
            if len(axsgroups_with_nax) == 0:
                continue

            Ms = symmetrize(Ms, axsgroups_with_nax)
            
    return Ms

def rounded(f):
    """Returns an integer if the float f is very close to one"""
    if np.round(f) == np.round(f, decimals=8):
        return int(np.round(f))
    else:
        return f
    
class IsotropicTensor():
    def __init__(self, numeric_rep, symbol=None, symbol_com=None, symshape=None, perm=None):
        """
        An isotropic Tensor is a tensor that is invariant under any coordinate rotations
        
        numeric_rep : A numpy matrix representation
        symbol : a sympy symbol
        symbol_com : a commutative symbol
        symshape : the symmetries of the tensor e.g. (2,2,2) (optional)
        perm : index permutation (optional)
        """
        
        self.numeric_rep = numeric_rep
        self.symb = symbol
        if symbol_com is None:
            symbol_com = symbol
        self.symb_com = symbol_com
        
        if symshape is None:
            rank = len(self.numeric_rep.shape)
            symshape = (rank//2, rank//2)
        self.symshape = symshape
        self.perm = perm
        
    def N(self):
        """Numeric Representation"""
        return self.numeric_rep
    
    def symbol(self, commutative=False):
        """A sympy symbol"""
        if commutative:
            return self.symb_com
        else:
            return self.symb
    
    def norm2(self):
        """Sum of all components squared"""
        return rounded(np.sum(self.numeric_rep*self.numeric_rep))
    
    def inner_product(self, other):
        """Full inner product with a tensor of the same shape"""
        if type(other) == type(self):
            on = other.N()
        else:
            on = other
        
        assert self.N().shape == on.shape, "Shape mismatch:" + str(self.N().shape) + str(on.shape)
        return rounded(np.sum(self.N()*on))
    
    def index_representation(self):
        assert ((self.perm is not None) and (self.symshape is not None)), "Unknown index representation"
        
        sym_str = ("%d" * len(self.symshape)) % self.symshape
        delta_str = ("\delta_{%s%s}" * (len(self.perm)//2)) % tuple(self.perm)
        
        return sympy.Symbol("S_{%s}(%s)" % (sym_str, delta_str))

def orthogonalize(basis, symbols=None):
    """Gram Schmidt Orthogonalization for Isotropic Tensors
    
    Input: n Isotropic Tensors I
    Output: n Isotropic Tensors J so that Ja * Jb = delta(a,b) Ja**2
    """
    newbasis = []

    for i in range(0, len(basis)):
        newt = basis[i].N()
        for j in range(0, len(newbasis)):
            a = basis[i].inner_product(newbasis[j]) / newbasis[j].norm2()
            newt = newt - a * newbasis[j].N()
        
        if symbols is not None:
            symbol = sympy.Symbol(symbols[i], commutative=False)
            symbol_com = sympy.Symbol(symbols[i], commutative=True)
        else:
            symbol = sympy.Symbol("X")
        
        newbasis.append(IsotropicTensor(newt, symbol=symbol,symbol_com=symbol_com, symshape=basis[i].symshape))
        
    return newbasis

def my_tensordot(t1, t2, s1, s2, axmul=(0,0)):
    """returns a tensorproduct over tensors with a symmetry condition
    
    t1, t2 : tensors
    s1, s2 : symmetry shapes
    axmul: The axes that should be contracted. Only contractions over axes with same symmetries are permitted
    
    returns t,s : new tensor, new symmetry shape
    """
    axmul = np.atleast_2d(axmul)

    snew = []
    
    offset = 0
    
    axs_tdot_a = []
    axs_tdot_b = []
    
    for i,s in enumerate(s1):
        if i in axmul[:,0]:
            axs_tdot_a.extend(offset + np.arange(s))
        else:
            snew.append(s)
        offset += s
    offset = 0
    for i,s in enumerate(s2):
        if i in axmul[:,1]:
            axs_tdot_b.extend(offset + np.arange(s))
        else:
            snew.append(s)
        offset += s
    
    res = np.tensordot(t1, t2, (axs_tdot_a, axs_tdot_b))
    
    return res, tuple(snew)

def mutli_dot(A, *Bs, products=()):
    """returns A multiple times multiplied and / or contracted with differnt B
    
    Assumes each axis of A is summed over exactly once
    """
    res, symshape = np.copy(A.N()), A.symshape
    
    offset = 0
    for i,B in enumerate(Bs):
        if products[i] == 2:
            res, symshape = my_tensordot(res, B.N(), symshape, B.symshape, axmul=((offset, 0), (offset+1, 1)))
        elif products[i] == 1:
            res, symshape = my_tensordot(res, B.N(), symshape, B.symshape, axmul=((offset, 0),))
            offset += 1
        else:
            raise ValueError("Can only handle contrs with 1s and 2s")
    return res, symshape

def pre_computed_invcov(derivs=(2,3,4)):
    if derivs == (2,3,4): # generated by sympy.srepr(its.pseudo_inverse_covariance_matrix(derivs=(2,3,4), full_output=True))
        strrep = "(MutableDenseMatrix([[Add(Mul(Integer(15), Pow(Symbol('sigma2', real=True, positive=True), Integer(2)), Pow(Add(Mul(Integer(2), Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Symbol('sigma2', real=True, positive=True), Integer(2))), Mul(Integer(-1), Integer(2), Pow(Symbol('sigma1', real=True, positive=True), Integer(4)))), Integer(-1)), Symbol('J_2=2', commutative=False)), Mul(Pow(Symbol('sigma2', real=True, positive=True), Integer(2)), Pow(Add(Mul(Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Symbol('sigma2', real=True, positive=True), Integer(2))), Mul(Integer(-1), Pow(Symbol('sigma1', real=True, positive=True), Integer(4)))), Integer(-1)), Symbol('J_22', commutative=False))), Integer(0), Add(Mul(Integer(15), Pow(Symbol('sigma1', real=True, positive=True), Integer(2)), Pow(Add(Mul(Integer(2), Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Symbol('sigma2', real=True, positive=True), Integer(2))), Mul(Integer(-1), Integer(2), Pow(Symbol('sigma1', real=True, positive=True), Integer(4)))), Integer(-1)), Symbol('J_2=4', commutative=False)), Mul(Pow(Symbol('sigma1', real=True, positive=True), Integer(2)), Pow(Add(Mul(Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Symbol('sigma2', real=True, positive=True), Integer(2))), Mul(Integer(-1), Pow(Symbol('sigma1', real=True, positive=True), Integer(4)))), Integer(-1)), Symbol('J_24', commutative=False)))], [Integer(0), Add(Mul(Integer(3), Pow(Symbol('sigma1', real=True, positive=True), Integer(-2)), Symbol('J_3-3', commutative=False)), Mul(Rational(35, 2), Pow(Symbol('sigma1', real=True, positive=True), Integer(-2)), Symbol('J_3\\\\equiv3', commutative=False))), Integer(0)], [Add(Mul(Integer(15), Pow(Symbol('sigma1', real=True, positive=True), Integer(2)), Pow(Add(Mul(Integer(2), Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Symbol('sigma2', real=True, positive=True), Integer(2))), Mul(Integer(-1), Integer(2), Pow(Symbol('sigma1', real=True, positive=True), Integer(4)))), Integer(-1)), Symbol('J_4=2', commutative=False)), Mul(Pow(Symbol('sigma1', real=True, positive=True), Integer(2)), Pow(Add(Mul(Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Symbol('sigma2', real=True, positive=True), Integer(2))), Mul(Integer(-1), Pow(Symbol('sigma1', real=True, positive=True), Integer(4)))), Integer(-1)), Symbol('J_42', commutative=False))), Integer(0), Add(Mul(Integer(15), Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Add(Mul(Integer(2), Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Symbol('sigma2', real=True, positive=True), Integer(2))), Mul(Integer(-1), Integer(2), Pow(Symbol('sigma1', real=True, positive=True), Integer(4)))), Integer(-1)), Symbol('J_4=4', commutative=False)), Mul(Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Add(Mul(Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Symbol('sigma2', real=True, positive=True), Integer(2))), Mul(Integer(-1), Pow(Symbol('sigma1', real=True, positive=True), Integer(4)))), Integer(-1)), Symbol('J_44', commutative=False)), Mul(Rational(315, 8), Pow(Symbol('sigma2', real=True, positive=True), Integer(-2)), Symbol('J_4==4', commutative=False)))]]), (Symbol('sigma0', real=True, positive=True), Symbol('sigma1', real=True, positive=True), Symbol('sigma2', real=True, positive=True)), {0: {0: [Mul(Pow(Symbol('sigma2', real=True, positive=True), Integer(2)), Pow(Add(Mul(Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Symbol('sigma2', real=True, positive=True), Integer(2))), Mul(Integer(-1), Pow(Symbol('sigma1', real=True, positive=True), Integer(4)))), Integer(-1))), Mul(Integer(15), Pow(Symbol('sigma2', real=True, positive=True), Integer(2)), Pow(Add(Mul(Integer(2), Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Symbol('sigma2', real=True, positive=True), Integer(2))), Mul(Integer(-1), Integer(2), Pow(Symbol('sigma1', real=True, positive=True), Integer(4)))), Integer(-1)))], 1: [], 2: [Mul(Pow(Symbol('sigma1', real=True, positive=True), Integer(2)), Pow(Add(Mul(Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Symbol('sigma2', real=True, positive=True), Integer(2))), Mul(Integer(-1), Pow(Symbol('sigma1', real=True, positive=True), Integer(4)))), Integer(-1))), Mul(Integer(15), Pow(Symbol('sigma1', real=True, positive=True), Integer(2)), Pow(Add(Mul(Integer(2), Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Symbol('sigma2', real=True, positive=True), Integer(2))), Mul(Integer(-1), Integer(2), Pow(Symbol('sigma1', real=True, positive=True), Integer(4)))), Integer(-1)))]}, 1: {0: [], 1: [Mul(Integer(3), Pow(Symbol('sigma1', real=True, positive=True), Integer(-2))), Mul(Rational(35, 2), Pow(Symbol('sigma1', real=True, positive=True), Integer(-2)))], 2: []}, 2: {0: [Mul(Pow(Symbol('sigma1', real=True, positive=True), Integer(2)), Pow(Add(Mul(Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Symbol('sigma2', real=True, positive=True), Integer(2))), Mul(Integer(-1), Pow(Symbol('sigma1', real=True, positive=True), Integer(4)))), Integer(-1))), Mul(Integer(15), Pow(Symbol('sigma1', real=True, positive=True), Integer(2)), Pow(Add(Mul(Integer(2), Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Symbol('sigma2', real=True, positive=True), Integer(2))), Mul(Integer(-1), Integer(2), Pow(Symbol('sigma1', real=True, positive=True), Integer(4)))), Integer(-1)))], 1: [], 2: [Mul(Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Add(Mul(Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Symbol('sigma2', real=True, positive=True), Integer(2))), Mul(Integer(-1), Pow(Symbol('sigma1', real=True, positive=True), Integer(4)))), Integer(-1))), Mul(Integer(15), Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Add(Mul(Integer(2), Pow(Symbol('sigma0', real=True, positive=True), Integer(2)), Pow(Symbol('sigma2', real=True, positive=True), Integer(2))), Mul(Integer(-1), Integer(2), Pow(Symbol('sigma1', real=True, positive=True), Integer(4)))), Integer(-1))), Mul(Rational(315, 8), Pow(Symbol('sigma2', real=True, positive=True), Integer(-2)))]}}, {0: {0: ['J22', 'J2=2'], 1: [], 2: ['J24', 'J2=4']}, 1: {0: [], 1: ['J3-3', 'J3---3'], 2: []}, 2: {0: ['J42', 'J4=2'], 1: [], 2: ['J44', 'J4=4', 'J4==4']}})"        
        return sympy.sympify(strrep)
    else:
        return None


class IsotropicTensorSystem():
    def __init__(self, maxorder=2, verbose=False, usecache=True):
        I2 = np.diag((1.,1.,1.))
        
        def create_tens(perm, symgroup, symbol="x", sym_equal_len_groups=True):
            ndelta = len(perm) // 2

            if ndelta == 1:
                delta_indices = "ij"
            elif ndelta == 2:
                delta_indices = "ij,kl"
            elif ndelta == 3:
                delta_indices = "ij,kl,mn"
            elif ndelta == 4:
                delta_indices = "ij,kl,mn,op"
            else:
                assert 0
            
            I2n = (I2,)*ndelta
            
            symb = sympy.Symbol(symbol, commutative=False)
            symb_com = sympy.Symbol(symbol, commutative=True)
            
            num = symmetrizeGroups(np.einsum("%s->%s" % (delta_indices, perm), *I2n), symgroup, sym_equal_len_groups=sym_equal_len_groups)
            
            symshape = tuple(len(s) for s in symgroup)
            
            return IsotropicTensor(num, symbol=symb, symbol_com=symb_com, symshape=symshape, perm=perm)
        
        self.sigma = sympy.symbols("sigma:3", real=True, positive=True)
        self.x = sympy.Matrix([sympy.symbols("T R S", commutative=False)]).T
        self.sigstar4 = self.sigma[0]**2*self.sigma[2]**2 - self.sigma[1]**4
        
        basis, obasis = {}, {}
        for bname in ("", "2", "4", "6", "8", "22", "33", "42", "24", "44", "222"):
            basis[bname], obasis[bname] = {}, {}
            
        # Fully symmetric tensors
        for empty_base in ("3", "5", "7"):
            basis[empty_base], obasis[empty_base] = {}, {}

        basis[""]["I0"] = IsotropicTensor(np.array(1), symbol=sympy.Number(1), symshape=(), perm="")
        basis["2"]["I2"] = create_tens("ij", ((0,1),), symbol="I_2")
        basis["4"]["I4"] = create_tens("ijkl", ((0,1,2,3),), symbol="I_4")
        basis["6"]["I6"] = create_tens("ijklmn", ((0,1,2,3,4,5),), symbol="I_6")
        basis["8"]["I8"] = create_tens("ijklmnop", ((0,1,2,3,4,5,6,7),), symbol="I_8")
        
        obasis[""]["J0"] = IsotropicTensor(np.array(1), symbol=sympy.Number(1), symshape=(), perm="")
        obasis["2"]["J2"] = create_tens("ij", ((0,1),), symbol="J_2")
        obasis["4"]["J4"] = create_tens("ijkl", ((0,1,2,3),), symbol="J_4")
        obasis["6"]["J6"] = create_tens("ijklmn", ((0,1,2,3,4,5),), symbol="J_6")
        obasis["8"]["J8"] = create_tens("ijklmnop", ((0,1,2,3,4,5,6,7),), symbol="J_8")

        symgroup = ((0,1), (2,3))
        basis["22"]["I22"] = create_tens("ijkl", symgroup, symbol="I_22")
        basis["22"]["I2=2"] = create_tens("ikjl", symgroup, symbol="I_2=2")
        obasis["22"]["J22"], obasis["22"]["J2=2"] = orthogonalize([basis["22"][k] for k in basis["22"]], symbols=("J_22", "J_2=2"))
        
        # 33
        symgroup = ((0,1,2), (3,4,5))
        basis["33"]["I3-3"] = create_tens("ijklmn", symgroup, symbol="I_3-3")
        basis["33"]["I3---3"] = create_tens("ikmjln", symgroup, symbol="I_3\equiv3")
        obasis["33"]["J3-3"], obasis["33"]["J3---3"] = orthogonalize([basis["33"][k] for k in basis["33"]], symbols=("J_3-3", "J_3\equiv3"))
        
        symgroup = ((0,1,2,3), (4,5))
        basis["42"]["I42"] = create_tens("ijklmn", symgroup, symbol="I_42")
        basis["42"]["I4=2"] = create_tens("ijkmln", symgroup, symbol="I_4=2")
        obasis["42"]["J42"], obasis["42"]["J4=2"] = orthogonalize([basis["42"][k] for k in basis["42"]], symbols=("J_42", "J_4=2"))
        symgroup = ((0,1), (2,3,4,5))
        basis["24"]["I24"] = create_tens("ijklmn", symgroup, symbol="I_24")
        basis["24"]["I2=4"] = create_tens("ikjlmn", symgroup, symbol="I_2=4")
        obasis["24"]["J24"], obasis["24"]["J2=4"] = orthogonalize([basis["24"][k] for k in basis["24"]], symbols=("J_24", "J_2=4"))
        symgroup = ((0,1,2,3), (4,5,6,7))
        basis["44"]["I44"] = create_tens("ijklmnop", symgroup, symbol="I_44")
        basis["44"]["I4=4"] = create_tens("ijkmlnop", symgroup, symbol="I_4=4")
        basis["44"]["I4==4"] = create_tens("ikmojlnp", symgroup, symbol="I_4==4")
        obasis["44"]["J44"], obasis["44"]["J4=4"], obasis["44"]["J4==4"] = orthogonalize([basis["44"][k] for k in basis["44"]], symbols=("J_44", "J_4=4", "J_4==4"))
        
        symgroup = ((0,1,2,3), (4,5))
        basis["42"]["I42"] = create_tens("ijklmn", symgroup, symbol="I_42")
        basis["42"]["I4=2"] = create_tens("ijkmln", symgroup, symbol="I_4=2")
        obasis["42"]["J42"], obasis["42"]["J4=2"] = orthogonalize([basis["42"][k] for k in basis["42"]], symbols=("J_42", "J_4=2"))
        symgroup = ((0,1), (2,3,4,5))
        basis["24"]["I24"] = create_tens("ijklmn", symgroup, symbol="I_24")
        basis["24"]["I2=4"] = create_tens("ikjlmn", symgroup, symbol="I_2=4")
        obasis["24"]["J24"], obasis["24"]["J2=4"] = orthogonalize([basis["24"][k] for k in basis["24"]], symbols=("J_24", "J_2=4"))
        symgroup = ((0,1,2,3), (4,5,6,7))
        basis["44"]["I44"] = create_tens("ijklmnop", symgroup, symbol="I_44")
        basis["44"]["I4=4"] = create_tens("ijkmlnop", symgroup, symbol="I_4=4")
        basis["44"]["I4==4"] = create_tens("ikmojlnp", symgroup, symbol="I_4==4")
        obasis["44"]["J44"], obasis["44"]["J4=4"], obasis["44"]["J4==4"] = orthogonalize([basis["44"][k] for k in basis["44"]], symbols=("J_44", "J_4=4", "J_4==4"))

        # 3 Tens
        symgroup = ((0,1), (2,3), (4,5))
        basis["222"]["I222"] = create_tens("ijklmn", symgroup, symbol="I_222")
        basis["222"]["I2=22"] = create_tens("ikjlmn", symgroup, symbol="I_2=22")
        basis["222"]["I2-2-2-"] = create_tens("iklmnj", symgroup, symbol="I_2-2-2-")
        obasis["222"]["J222"], obasis["222"]["J2=22"], obasis["222"]["J2-2-2-"] = orthogonalize([basis["222"][k] for k in basis["222"]], symbols=("J_222", "J_2=22", "J_2-2-2-"))
        
        #symgroup = ((0,1), (2,3), (4,5,6,7))
        #basis["224"]["I224"] = create_tens("ijklmnop", symgroup, symbol="I_224")
        #basis["224"]["I2=24"] = create_tens("ikjlmnop", symgroup, symbol="I_2=24")
        #basis["224"]["I22=4"] = create_tens("ijkmlnop", symgroup, symbol="I_22=4")
        #basis["224"]["I2-2-4-"] = create_tens("iklmnjop", symgroup, symbol="I_2-2-4-")
        #basis["224"]["I22=4="] = create_tens("iojpkmln", symgroup, symbol="I_22=4=")
        #obasis["224"]["I224"], obasis["224"]["I2=24"], obasis["224"]["I22=4"], obasis["224"]["I2-2-4-"], obasis["224"]["I22=4="] = orthogonalize([basis["224"][k] for k in basis["224"]], symbols=("J_224", "J_2=24", "J_22=4", "J_2-2-4-", "J_22=4="))

        
        self.basis = basis
        self.obasis = obasis
        
        self.tens, self.otens = {}, {}
        for bname in basis:
            self.otens.update(self.obasis[bname])
            self.tens.update(self.basis[bname])
        self.tens.update(self.otens)
        #self.otens = {**self.otens, *[**obasis[bname] for bname in self.obasis]}
        
        
        #self.otens = {**obasis[""], **obasis["2"], **obasis["22"], **obasis["222"], **obasis["33"], **obasis["42"], **obasis["24"], **obasis["44"], **obasis["4"], **obasis["6"], **obasis["8"]}
        #self.tens = {**basis[""], **basis["2"], **basis["22"], **basis["222"], **basis["33"], **basis["42"], **basis["24"], **basis["44"], **basis["4"], **basis["6"], **basis["8"], **self.otens}
        
        self.orth_algebra_2 = self.symbol_algebra(product=2, keepzero=True, keepsymmetric=True)
        self.orth_algebra_3 = self.symbol_algebra(product=3, keepzero=True, keepsymmetric=True)
        self.orth_algebra_4 = self.symbol_algebra(product=4, keepzero=True, keepsymmetric=True)
        
        self.orth_algebra = (*self.orth_algebra_2, *self.orth_algebra_3, *self.orth_algebra_4)
        
        self.icov_cache = {}
        if usecache:
            self.icov_cache[(2,3,4)] = pre_computed_invcov((2,3,4))
        
        
        
    def symbol_ItoJ(self, basis_name=None, limit_denom=10**8):
        """Can be used to switch from non-otrthognal tensor symbols I to orthogonal ones J"""
        if basis_name is not None:
            nItoJ = self.basis_relation(basis_name)

            res = []
            for i,t1 in enumerate(self.basis[basis_name]):
                val = 0
                for j,t2 in enumerate(self.obasis[basis_name]):
                    val = val + sympy.Rational(nItoJ[i,j]).limit_denominator(limit_denom) * self(t2).symbol()

                res.append((self(t1).symbol(), val))
            return res
        else:
            res = []
            for bname in self.basis:
                res.extend(self.symbol_ItoJ(bname, limit_denom=limit_denom))
            return res
                
    
    def symbol_JtoI(self, basis_name=None, limit_denom=10**8, get_name=False, index_rep=False):
        """Can be used to switch from otrthognal tensor symbols I to non-orthogonal ones J"""
        if basis_name is not None:
            nItoJ = self.basis_relation(basis_name, inverse=True)

            res = []
            for i,t1 in enumerate(self.obasis[basis_name]):
                val = 0
                for j,t2 in enumerate(self.basis[basis_name]):
                    if index_rep:
                        st2 = self(t2).index_representation()
                    else:
                        st2 = self(t2).symbol()
                    val = val + sympy.Rational(nItoJ[i,j]).limit_denominator(limit_denom) * st2

                if get_name:
                    res.append((t1,self(t1).symbol(), val))
                else:
                    res.append((self(t1).symbol(), val))
            return res
        else:
            res = []
            for bname in self.obasis:
                res.extend(self.symbol_JtoI(bname, limit_denom=limit_denom, get_name=get_name, index_rep=index_rep))
            return res
        
    def symbol_decompose(self, tens, basis_name, limit_denom=10**8):
        """Symbolicly decompose an isotropic tensor into an orthogonal basis
        
        e.g. 
        its = IsotropicTensorSystem()
        its.symbol_decompose(its("I2=2"), "22")
        returns J22/3 + J2=2
        """
        if(type(tens) == IsotropicTensor):
            num = tens.N()
        else:
            num = tens
        
        resN = self.decompose(num, basis_name)
        
        res = 0
        for i,tname in enumerate(self.obasis[basis_name]):
            res += sympy.Rational(resN[i]).limit_denominator(limit_denom) * self(tname).symbol()
        return res
    
    def decompose(self, tens, basis_name, verbose=False):
        """Decompose an isotropic tensor into an orthogonal basis
        
        e.g. 
        its = IsotropicTensorSystem()
        its.decompose(its.N("I2=2"), "22")
        returns [1/3, 1]
        """
        basis = self.obasis[basis_name]
        
        res = []
        remain = np.copy(tens)
        for tname in basis:
            if basis[tname].norm2() <= 1e-10:
                res.append(0)
            else:
                res.append(basis[tname].inner_product(tens) / basis[tname].norm2())
                remain = remain - res[-1] * basis[tname].N()
        if verbose:
            print(remain)

        return np.array(res)
    
    def basis_relation(self, basis_name, inverse=False):
        """The Matrix M_ItoJ so that
        cI = M_ItoJ * cJ
        or the inverse transformation
        
        e.g. basis_relation("22")
        [[1, 0]
         [1/3, 1]]
        basis_relation("22", inverse=True)
        [[1, 0]
         [-1/3, 1]]
        """
        ItoJ = np.array([self.decompose(self.N(k), basis_name) for k in self.basis[basis_name]])
        
        if inverse & (len(ItoJ) > 0):
            return np.linalg.inv(ItoJ)
        else:
            return ItoJ
        
    def print_bases(self, inverse=False):
        for bname in self.basis.keys():
            rel = self.basis_relation(bname, inverse=inverse)
            print("---", bname, "---")
            print(rel)
        

    def __call__(self, name):
        """Returns a tensor with given name e.g. 'J22'"""
        return self.tens[name]
    
    def N(self, name):
        """Numeric representation of a tensor"""
        return self.tens[name].N()
    
    def basis_name(self, symshape):
        """Returns the name of the basis with given symmetry-groups
        e.g.
        basis_name((3,3)) = "33"
        """
        name = ""
        for s in symshape:
            name += str(s)
        return name
    
    def symbol_algebra(self, product=2, basis_name=None, keepzero=False, keepsymmetric=False, keepnonsense=False, keepmissingbasis=False, limit_denom=10**8):
        """Returns a symbolic algebra of simple inner products.
        This can be used with sympy.subs(algebra) to simplify terms
        with products of different Isotropic tensors
        """
        if basis_name is None:
            leftside = rightside = self.otens
        else:
            leftside = rightside = self.obasis[basis_name]
        
        res = []
        for i,t1 in enumerate(leftside):
            for j,t2 in enumerate(rightside):
                if (not keepsymmetric) & (j < i):
                    continue
                
                sym1, sym2 = self(t1).symshape, self(t2).symshape
                if (len(sym1) == 0) | (len(sym2) == 0):
                    continue
                
                if (sym1[-1] == product) & (sym2[0] == product):
                    newt = np.tensordot(self.N(t1), self.N(t2), product)

                    ressym = sym1[:-1] + sym2[1:]

                    if len(ressym) > 0:
                        if self.basis_name(ressym) in self.obasis:
                            sym_newt = self.symbol_decompose(newt, self.basis_name(ressym))
                        else:
                            if keepmissingbasis:
                                sym_newt = np.nan
                            else:
                                continue
                    else: # We have a scalar
                        sym_newt = sympy.Rational(float(newt)).limit_denominator(limit_denom)
                else: # The product does not make sense, just set it to zero
                    #print("skip", t1, t2, sym1, sym2)
                    if keepnonsense:
                        sym_newt = 0
                    else:
                        continue

                if (sym_newt == 0) & (not keepzero):
                    continue

                res.append((self(t1).symbol()*self(t2).symbol(), sym_newt))
        
        return res
    
    def covariance_matrix(self,derivs=(2,),decomposed=True):  
        """The covariance matrix of a given list of derivatives of the potential
        If decomposed is set to True, it will be
        decomposed into Isotropic Tensors with symmetry deriv x deriv
        
        E.g. for derivs = (2,)
        this gives the covariance matrix of the tidal tensor.
        E.g. for derivs = (2,3,4)
        this gives the (block) joint-covariance matrix of tidal tensor, third and fourth derivatives of the potential        
        """
        C = sympy.Matrix(np.zeros((len(derivs),len(derivs))))
        
        sigma = self.sigma

        for i,d1 in enumerate(derivs):
            for j,d2 in enumerate(derivs):
                rank = d1+d2
                sigi = sigma[rank//2 - 2]
                
                if rank % 2 == 0:
                    sign = (-1)**(np.abs(d1-d2)//2)
                else:
                    continue # Uneven terms are always 0

                if decomposed:
                    basis_name = self.basis_name((d1,d2))
                    C[i,j] = (sign *self.symbol_decompose(self("J%d" % (d1+d2)), basis_name) * sigi**2 / (rank + 1)).expand()
                else:
                    C[i,j] = self("J%d" % (d1+d2)).symbol() * sigi**2 / (rank + 1)
        return C, sigma
    
    def pseudo_inverse_covariance_matrix(self, derivs=(2,), generic=False, full_output=False):
        """The pseudo inverse C+ of the covariance matrix. It can be inferred through
        C = C C+ C
        
        by default returns C+, sigma
        if full_output:
        returns C+, sigma, As, Js
        where As are coefficients of tensors Js that can be used to reconstruct
        the inverse covariance matrix.
        """
        
        if (not generic) & (derivs in self.icov_cache):
            Cires, sigma, As, Js = self.icov_cache[derivs]
            if full_output:
                return Cires, sigma, As, Js
            else:
                return Cires, sigma
                
        
        Ci = sympy.Matrix(np.zeros((len(derivs),len(derivs))))
        if generic:
            As = {}
            As_flat = []
            Js = {}
            for i,d1 in enumerate(derivs):
                As[i] = {}
                Js[i] = {}
                for j,d2 in enumerate(derivs):
                    As[i][j] = []
                    Js[i][j] = []
                    rank = d1+d2

                    basis_name = self.basis_name((d1,d2))
                    if not basis_name in self.obasis:
                        continue
                    
                    for k,tname in enumerate(self.obasis[basis_name]):
                        if i <= j:
                            A = sympy.symbols("A%s" % str(self(tname).symbol())[1:], real=True)
                            As_flat.append(A)
                        else:
                            A = As[j][i][k]
                        As[i][j].append(A)
                            
                        
                        Ci[i,j] += A * self(tname).symbol()
                        Js[i][j].append(tname) # self(tname).symbol()
            return Ci, As_flat, As, Js
        else:
            C, sigma = self.covariance_matrix(derivs, decomposed=True)
            Ci, A, As, Js = self.pseudo_inverse_covariance_matrix(derivs, generic=True)
            CCi = (C*Ci).expand().subs(self.orth_algebra).expand()
            CCiC = (CCi*C).expand().subs(self.orth_algebra).expand()
            
            # We have to solve the Equation
            # CCiC = C
            # To make this easier for sympy, we separate by coefficients
            # since the coefficients of these all have to match:
            Eq = []
            for i,d1 in enumerate(derivs):
                for j,d2 in enumerate(derivs):
                    #if j<i:
                    #    continue
                    for Jname in Js[i][j]:
                        J = self(Jname).symbol()
                        Eq.append(CCiC[i,j].coeff(J) - C[i,j].coeff(J))
            sol = sympy.solve(Eq, A, dict=True)
            
            Cires = Ci.subs(sol[0])

            for i in As:
                for j in As[i]:
                    for k in range(0, len(As[i][j])):
                        As[i][j][k] = sol[0][As[i][j][k]]
            
            self.icov_cache[derivs] = Cires, sigma, As, Js
            
            if full_output:
                return Cires, sigma, As, Js
            else:
                return Cires, sigma
        
    def symbol_transpose(self, term):
        """Transposes 42 terms into 24 and vice versa"""
        l42, l24 = [],[]
        
        for tname in self.obasis["42"]:
            l42.append(self(tname).symbol())
        for tname in self.obasis["24"]:
            l24.append(self(tname).symbol())
            
        dummy = sympy.symbols("dummy:%d" % len(l24))

        orig = [*l24, *l42, *dummy]
        transpose = [*dummy, *l24, *l42]
        subs = tuple(zip(orig,transpose))
        
        return term.subs(subs)
    
    def symbol_apply_algebra(self, term, nsimp=1):
        """Applies the simple inner product algebra to a sympy expression"""
        if nsimp==0:
            return term
        else:
            return sympy.simplify(self.symbol_apply_algebra(term, nsimp-1).expand().subs(self.orth_algebra))
        
    def symbol_collect_Js(self, term):
        """Collect Js in a sympy expression"""
        allJs = [self(tname).symbol() for tname in self.otens]
        allJscom = [self(tname).symbol(commutative=True) for tname in self.otens]
        
        return term.subs(zip(allJs, allJscom)).collect(allJscom)#.subs(zip(allJscom, allJs))
    
    def coeffrep_to_symbol(self, coeff, sym):
        res = 0
        for c,tname in zip(coeff, self.obasis[self.basis_name(sym)]):
            res += c*self(tname).symbol()
        return res
        
    def symbol_bias_deriv1(self, term=None, potderiv=(2,3,4)):
        """This term (f) is defined so that b1* = < f >_galaxies
        
        term can e.g. be "J2". If term=None we return a dictionary with all possible terms
        
        f is given by
        (-1)^n  J * p^(n)/p / (J*J)
        The first derivative of a MV Gaussian is
        p'/p = -C*x
        So we calculate  J*C*x /(J*J) here
        """
        
        Ci, sigma, As, Js = self.pseudo_inverse_covariance_matrix(potderiv, full_output=True)

        x = self.x
        
        nderiv = len(potderiv)
        
        Ci_coeffrep = []
        for i in range(0,nderiv):
            Ci_coeffrep.append([])
            for j in range(0,nderiv):
                Ci_coeffrep[i].append((As[i][j], (potderiv[i], potderiv[j])))
        
        def single_term(term, component):
            res = 0
            for i in range(0, nderiv):
                crep = self.coeffrep_multidot(self(term), Ci_coeffrep[component][i], products=(1,))

                res += self.coeffrep_to_symbol(*crep) * x[i] / self(term).norm2()
            return sympy.together(res)

        res = {}
        if term is None: # return all terms
            v = Ci*x
            for comp, deriv in enumerate(potderiv):
                for tname in self.obasis["%d" % deriv]:
                    res[tname] = single_term(tname, comp)
            return res
        else:
            for comp, deriv in enumerate(potderiv):
                if term in self.obasis["%d" % deriv]:
                    return single_term(term, comp)
            return 0
        
    def symbol_bias_deriv2(self, term=None, potderiv=(2,3,4)):
        """This term (f) is defined so that b2* = < f >_galaxies
        
        term can e.g. be "J2=2". If term=None we return a dictionary with all possible terms
        
        f is given by
        (-1)^n  J * p^(n)/p / (J*J)
        The second derivative of a MV Gaussian is
        p''/p = (Cx) otimes (Cx) - C
        So we calculate  ((Cx).T J (Cx) - C J) /(J*J) here
        """
        # Bias terms are given by
        # (-1)^n  J * p^(n)/p / (J*J)
        # The second derivative of a MV Gaussian is
        # p''/p = (Cx) otimes (Cx) - C
        # So we calculate  ((Cx).T J (Cx) - C J) /(J*J) here
        
        Ci, sigma, As, Js = self.pseudo_inverse_covariance_matrix(potderiv, full_output=True)
        
        x = self.x
        
        nderiv = len(potderiv)
        
        Ci_coeffrep = []
        for i in range(0,nderiv):
            Ci_coeffrep.append([])
            for j in range(0,nderiv):
                Ci_coeffrep[i].append((As[i][j], (potderiv[i], potderiv[j])))
        
        def single_term(term, c1, c2):
            res = 0
            norm2 = sympy.Rational(self(term).norm2()).limit_denominator(10**8)
            for i in range(0, nderiv):
                for j in range(0, nderiv):
                    crep = self.coeffrep_multidot(self(term), Ci_coeffrep[c1][i], Ci_coeffrep[c2][j], products=(1,1))
                    res += self.coeffrep_to_symbol(*crep) * x[i] * x[j] / norm2
                    
            crep = self.coeffrep_multidot(self(term), Ci_coeffrep[c1][c2], products=(2,))
            res += - self.coeffrep_to_symbol(*crep) / norm2
            
            return sympy.together(res)

        res = {}
        if term is None: # return all terms
            v = Ci*x
            for c1, deriv1 in enumerate(potderiv):
                for c2, deriv2 in enumerate(potderiv):
                    if c1 > c2: # skip duplicate symmetric terms
                        continue
                    bname = "%d%d" % (deriv1, deriv2)
                    if not bname in self.obasis:
                        continue
                    for tname in self.obasis[bname]:
                        res[tname] = single_term(tname, c1, c2)
            return res
        else:
            for c1, deriv1 in enumerate(potderiv):
                for c2, deriv2 in enumerate(potderiv):
                    bname = "%d%d" % (deriv1, deriv2)
                    if not bname in self.obasis:
                        continue
                    if term in self.obasis[bname]:
                        return single_term(term, c1, c2)
            return 0
        
    def symbol_bias_deriv3(self, term, potderiv=(2,3,4)):
        """This term (f) is defined so that b3* = < f >_galaxies

        term can e.g. be "J2-2-2-". So far this method only works with potderiv=(2,) i.e.
        without spatial derivative corrections to the third order terms
        
        Where f is given by
        (-1)^n  J * p^(n)/p / (J*J)
        The third derivative of a MV Gaussian is
        p'''/p = 2 C otimes (Cx) - (Cx) otimes (Cx) otimes (Cx)
        So we calculate  ((Cx) otimes (Cx) otimes (Cx) J - 2 C otimes (Cx) J) /(J*J) here
        """
        Ci, sigma, As, Js = self.pseudo_inverse_covariance_matrix(potderiv, full_output=True)
        
        x = self.x
        
        nderiv = len(potderiv)
        
        Ci_coeffrep = []
        for i in range(0,nderiv):
            Ci_coeffrep.append([])
            for j in range(0,nderiv):
                Ci_coeffrep[i].append((As[i][j], (potderiv[i], potderiv[j])))
        
        def single_term(term, c1, c2, c3):
            res = 0
            norm2 = sympy.Rational(self(term).norm2()).limit_denominator(10**8)
            for i in range(0, nderiv):
                for j in range(0, nderiv):
                    for k in range(0, nderiv):
                        crep = self.coeffrep_multidot(self(term), Ci_coeffrep[c1][i], Ci_coeffrep[c2][j], Ci_coeffrep[c3][k], products=(1,1,1))
                        res += self.coeffrep_to_symbol(*crep) * x[i] * x[j] * x[k] / norm2
                    
                crep = self.coeffrep_multidot(self(term), Ci_coeffrep[c1][i], Ci_coeffrep[c2][c3], products=(1,2))
                res += - 3 * self.coeffrep_to_symbol(*crep) * x[i] / norm2
            
            return sympy.together(res)

        for c1, deriv1 in enumerate(potderiv):
            for c2, deriv2 in enumerate(potderiv):
                 for c3, deriv3 in enumerate(potderiv):
                    bname = "%d%d%d" % (deriv1, deriv2, deriv3)
                    if not bname in self.obasis:
                        continue
                    if term in self.obasis[bname]:
                        return single_term(term, c1, c2, c3)
        return np.nan
    
    def symbol_bias_derivN(self, term, potderiv=(2,3,4), getderiv=False):
        if term in ("J2","J4"):
            deriv, term = 1, self.symbol_bias_deriv1(term, potderiv=potderiv)
        elif term in ("J22", "J2=2", "J3-3", "J3---3", "J44", "J4=4", "J4==4", "J24", "J2=4"):
            deriv, term = 2, self.symbol_bias_deriv2(term, potderiv=potderiv)
        elif term in ("J222", "J2=22", "J2-2-2-"):
            deriv, term = 3, self.symbol_bias_deriv3(term, potderiv=potderiv)
        else:
            raise NotImplementedError(term)
        if getderiv:
            return term, deriv
        else:
            return term
        
        
    def coeffrep_dot(self, cA, cB, sA, sB, axmul=(0,0), limit_denom=10**8):
        """Evaluates the tensor product given a coefficient representation

        cA: coefficients of A
        cB: coefficients of A
        sA: symmetry of A. It is assumed that A = cA*basis(sA)
        sb: symmetry of B. It is assumed that B = cB*basis(sB)

        axmul: axes to multiply, can be (axA, axB) or ((axA1, axB1), (axA2, axB2), ...)

        returns newcoeff, newbasisname, newsym
        """
        if (len(cA) == 0) or (len(cB) == 0):
            return [], ()
        
        tnamesA = tuple(self.obasis[self.basis_name(sA)].keys())
        tnamesB = tuple(self.obasis[self.basis_name(sB)].keys())

        sym = None
        for i in range(0, len(cA)):
            for j in range(0, len(cB)):
                if((cA[i] == 0) | (cB[j] == 0)):
                    continue
                JcA, JcB = self(tnamesA[i]), self(tnamesB[j])

                res, newsym = my_tensordot(JcA.N(), JcB.N(), sA, sB, axmul=axmul)

                if sym is None:
                    sym = newsym
                    newbasis = self.basis_name(sym)
                    new_coeff = [0] * len(self.obasis[newbasis].keys())
                else:
                    assert sym == newsym

                res_decomp = self.decompose(res, newbasis)
                for c in range(0, len(new_coeff)):
                    new_coeff[c] += sympy.Rational(res_decomp[c]).limit_denominator(limit_denom) * cA[i] * cB[j]
        if sym is None: #didn't have a single non-zero case
            return (0,), ()            
        else:
            return new_coeff, sym
    
    def coeffrep_multidot(self, A, *crepBs, products=()):
        """returns A multiple times multiplied and / or contracted with differnt B

        Assumes each axis of A is summed over exactly once
        """
        if isinstance(A, IsotropicTensor):
            sA = A.symshape
            cA = self.decompose(A.N(), self.basis_name(sA))
            cA = [sympy.Rational(c).limit_denominator(10**8) for c in cA]
        else: # given in coefficient representation
            cA, sA = A

        offset = 0
        for i,(cB,sB) in enumerate(crepBs):
            if products[i] == 2:
                #print(((offset, 0), (offset+1, 1)))
                cA, sA = self.coeffrep_dot(cA, cB, sA, sB, axmul=((offset, 0), (offset+1, 1)))
            elif products[i] == 1:
                cA, sA = self.coeffrep_dot(cA, cB, sA, sB, axmul=((offset, 0),))
                #offset += 1
            else:
                raise ValueError("Can only handle contrs with 1s and 2s")
        return cA, sA
    
    def symbol_symsubs(self):
        subs = {}
        for t1,t2 in zip(self.obasis["42"].keys(), self.obasis["24"].keys()):
            subs[self(t1).symbol()] = self(t2).symbol()
        return subs

    def pycode_bias_estimator(self, term, potderiv=(2,3,4), label="o4", getinfo=False):
        import re

        t,deriv = self.symbol_bias_derivN(term, potderiv=potderiv, getderiv=True)
        
        prelabel = "deriv%d" % deriv
        
        #t = self.symbol_bias_deriv2(term, potderiv=potderiv)
        t = t.subs(zip(self.x, (1,1,1))).subs(self.symbol_symsubs())
        
        # This symbol is giving errors with regular expressions:
        t = t.subs(sympy.Symbol("J_3\equiv3", commutative=False), sympy.Symbol("J3---3"))

        symb = t.subs(zip(self.sigma, (1,0.1,1))).free_symbols
        
        pycode = "    return " + sympy.printing.pycode(t)
        pycode = re.sub("sigma0", "sigma[0]", pycode)
        pycode = re.sub("sigma1", "sigma[1]", pycode)
        pycode = re.sub("sigma2", "sigma[2]", pycode)

        symb_str = [sympy.printing.pycode(s) for s in symb]
        symb_strout = {}
        for s in symb_str:
            symb_strout[s] = 't["%s"]' % re.sub("_", "", s)
        symb_strout["J_22"] = 't["J2"]**2'
        symb_strout["J_44"] = 't["J4"]**2'
        symb_strout["J_24"] = symb_strout["J_42"] = 't["J2"]*t["J4"]'
        symb_strout["J_2=22"] = 't["J2=2"]*t["J2"]'
        
        for s in symb_str:
            pycode = re.sub(s, symb_strout[s], pycode) # 't["%s"]' % re.sub("_", "", s)

        termstr = re.sub("=", "__", term)
        termstr = re.sub("-", "_", termstr)
        
        funcname = "%s_%s_%s" % (prelabel, termstr, label)
        pycode = "def %s(t, sigma):\n" % (funcname) + pycode
        
        if getinfo:
            return pycode, funcname
        else:
            return pycode