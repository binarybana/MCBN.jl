
export BayesNet, BayesNetSampler

type Node 
    index::Int
    label::UTF8String
    var::Var
    fac::Factor
    function Node(i,l,v,f)
        assert(i==label(v))
        assert(v in vars(f))
        new(i,l,v,f)
    end
end

function show(io::IO, n::Node)
    try
        if int(n.label) == n.index
            print(io, "$(n.index)")
        end
    catch
        print(io, "$(n.index): $(n.label)")
    end
    print(io, " : [")
    varvec = vars(vars((n.fac)))
    for v in varvec
        print(io, "$(label(v)),")
    end
    print(io, "\b] : [")
    for i in 1:nrStates(n.fac)
        @printf io "%.2f," n.fac[i]
    end
    print(io, "\b]")
end

function make_vertex(g::AdjacencyList{Node}, x) 
    ind = num_vertices(g)+1
    v = Var(ind,2) #FIXME hardcoded binary
    Node(ind, string(x), v, Factor(v)) 
end
vertex_index(n::Node) = n.index

typealias BayesNet AdjacencyList{Node}

function BayesNet(n::Int)
    BayesNet(1:n)
end

function BayesNet(itr)
    g = adjlist(Node)
    for n in itr
        add_vertex!(g,n)
    end
    g
end

function add_edge!(g::AdjacencyList{Node}, u::Int, v::Int)
    un,vn = g.vertices[[u,v]]
    vn.fac = embed(vn.fac, vars(vn.fac)+u.var)
    add_edge!(g, un, vn)
end

function set_factor!(bnd::BayesNetDAI, i::Int, 

type BayesNetDAI
    bn::BayesNet
    fg::FactorGraph
    jt::JTree
    dirty::Bool
    memo_entropy::Float64
end

function BayesNetDAI(n::Int)
    g = BayesNet(n)
    BayesNetDAI(g)
end

function BayesNetDAI(bn::BayesNet)
    fg = FactorGraph([x.fac for x in bn.vertices]...)
    jt = JTree(fg)
    BayesNetDAI(bn, fg, jt, true, -Inf)
end

type BayesNetSampler
    bn::BayesNetDAI
    data::Matrix{Int}
    cpds
    template
end

#propose
#energy
#reject
 
function kld(bnd::BayesNetDAI, other::BayesNetDAI)
    if !bnd.dirty
        bnd.memo_entropy = entropy(bnd)
        # Now entropy and jtree are correct
    end
    accum = 0.0
    for i=1:numFactors(bnd.fg)
        v = other.bn.vertices[i].var
        arity = states(v)
        numparstates = iround(nrStates(other.fg[i])/arity)
        # Assumption: same ordering between bnd and other
        jointvars = vars(other.fg[i])
        parvars = jointvars - v
        margfac = marginal(bnd.jt, parvars)
        joint = marginal(bnd.jt, jointvars)
        for parstate=1:numparstates
            subaccum = 0.0
            for state=1:arity
                index = conditionalState(v, parvars, state, parstate)
                p_x_par = other.fg[i][index]
                q_condprob = joint[index]/margfac[parstate]
                subaccum += q_condprob * log2(p_x_par)
            end
            accum += margfac[parstate] * subaccum
        end
    end
    return -bnd.memo_entropy - accum
end
 
function faster_kld(bnd::BayesNetDAI, other::BayesNetDAI)
    if !bnd.dirty
        bnd.memo_entropy = entropy(bnd)
        # Now entropy and jtree are correct
    end
    accum = 0.0
    for i=1:numFactors(bnd.fg)
        v = other.bn.vertices[i].var
        arity = states(v)
        numparstates = iround(nrStates(other.fg[i])/arity)
        # Assumption: same ordering between bnd and other
        jointvars = vars(other.fg[i])
        parvars = jointvars - v
        joint = marginal(bnd.jt, jointvars)
        for parstate=1:numparstates
            for state=1:arity
                index = conditionalState(v, parvars, state, parstate)
                p_x_par = other.fg[i][index]
                accum += joint[index] * log2(p_x_par)
            end
        end
    end
    return -bnd.memo_entropy - accum
end

function entropy(bnd::BayesNetDAI)
    if !bnd.dirty
        return bnd.memo_entropy
    end
    init!(bnd.jt)
    run!(bnd.jt)
    bnd.dirty = false
    accum = 0.0
    for i in 1:numFactors(bnd.fg)
        v = bnd.bn.vertices[i]
        arity = states(v.var)
        numparstates = iround(nrStates(bnd.fg[i])/arity)
        pars = vars(bnd.fg[i]) - v.var
        margfac = marginal(bnd.jt, pars)
        for parstate in 1:numparstates
            accumsum = 0.0
            for state in 1:arity
                temp = bnd.fg[i][conditionalState(v.var, pars, state, parstate)]
                accumsum -= temp == 0.0 ? 0.0 : temp * log2(temp)
            end
            accum += margfac[parstate] * accumsum
        end
    end
    bnd.memo_entropy = accum
    return accum
end

function naive_entropy(bnd::BayesNetDAI)
    numstates = 1
    accum = 0.0
    allvars = VarSet([x.var for x=bnd.bn.vertices]...)
    for n in bnd.bn.vertices
        numstates *= states(n.var)
    end

    for i=1:numstates
        accumprod = 1.0
        statemap = calcState(allvars, i)
        for j in 1:numFactors(bnd.fg)
            reducedstate = calcLinearState(vars(bnd.fg[j]), statemap, allvars)
            accumprod *= bnd.fg[j][reducedstate]
        end
        accum += accumprod == 0 ? 0.0 : accumprod * log2(accumprod)
    end
    return -accum
end


#
#need to handle prior information: structure and cpd
#init cpds
#init structures

