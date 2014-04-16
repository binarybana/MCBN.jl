export BayesNet, BayesNetSampler

type Node 
    index::Int
    label::UTF8String
    var::Var
    function Node(i,l,v)
        assert(i==label(v))
        new(i,l,v)
    end
end
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

function show(io::IO, n::Node)
    try
        if int(n.label) == n.index
            print(io, "$(n.index)")
        end
    catch
        print(io, "$(n.index): $(n.label)")
    end
end

function make_vertex(g::AdjacencyList{Node}, x) 
    ind = num_vertices(g)+1
    v = Var(ind,2) #FIXME hardcoded binary
    Node(ind, string(x), v) 
end
vertex_index(n::Node) = n.index

function adjmat(bn::BayesNet)
    n = length(bn.vertices)
    mat = zeros(n,n)
    for i in 1:n, neigbor=bn.adjlist[i]
        mat[i,vertex_index(neighbor)] = 1
    end
    mat
end

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
    fg = FactorGraph([Factor(x.var) for x in bn.vertices]...)
    jt = JTree(fg)
    BayesNetDAI(bn, fg, jt, true, -Inf)
end

function add_edge!(bnd::BayesNetDAI, u::Int, v::Int)
    un,vn = bnd.bn.vertices[[u,v]]
    vnfac = bnd.fg[v]
    bnd.dirty=true
    bnd.fg[v] = embed(vnfac, vars(vnfac)+un.var)
    add_edge!(bnd.bn, un, vn)
end

function set_factor!(bnd::BayesNetDAI, i::Int, vals::Vector{Float64})
    bnd.dirty = true
    fcopy = bnd.fg[i]
    assert(length(vals) == nrStates(fcopy))
    oldvals = p(fcopy)
    oldvals[:] = vals
    bnd.fg[i] = fcopy
end

function show(io::IO, bnd::BayesNetDAI)
    show(io, bnd.bn)
    show(io, bnd.fg)
    #print(io, " : [")
    #varvec = vars(vars((n.fac)))
    #for v in varvec
        #print(io, "$(label(v)),")
    #end
    #print(io, "\b] : [")
    #for i in 1:nrStates(n.fac)
        #@printf io "%.2f," n.fac[i]
    #end
    #print(io, "\b]")
end

type BayesNetSampler
    bnd::BayesNetDAI
    data::Matrix{Int}
    p_cpds::Float64
    cpds
    p_structural::Float64
    template
    limparent::Int
    changelist::Vector{Int}
    x::Vector{Int} # node ordering/mapping
    fvalue::Vector{Float64}
    logqfactor::Float64
end

BayesNetSampler(n::Int, data::Matrix{Int}) = BayesNetSampler(
    BayesnetDAI(n),
    data,
    0.2,
    [],
    0.2,
    BayesNet(n),
    4,
    [1:n],
    shuffle([1:n]),
    zeros(Float64, n),
    0.0)

function factor_klds(bnd1::BayesNetDAI, bnd2::BayesNetDAI)
    0
end

function energy(bns::BayesNetSampler)
    # energy contribution from structural
    adj_diff = sum(abs(adjmat(bns.bnd.bn) - adjmat(bns.template)))
    e_struct = bns.p_structural * adj_diff

    # energy contribution from cpds
    cpd_diff = factor_klds(bns.bnd, bns.cpds)
    e_cpd = bns.p_cpds * cpd_diff

    ##### energy from data (likelihood) ####
    accum = 0.0

    #limit number of parents
    if bns.bnd.bn.nedges > bns.limparent
        return 1e20
    end

    counts = {}
    for c in bns.changelist
        node = bns.x[c]
        push!(counts, zeros(Int,nrStates(bns.fg[node])))
    end

    allvars = VarSet([x.var for x=bnd.bn.vertices]...)
    for i=1:length(counts)
        node = bns.x[bns.changelist[i]]
        fac = bns.bnd.fg[node]
        facvs = vars(fac)
        inds = labels(fac)
        # TODO check indexing above, do
        # I need to pass inds through bns.x again?
        for row=size(bns.data,1)
            index = calcLinearState(facvs, bns.data[inds]) 
            counts[node][index] += 1
        end
    end

    for i=1:length(counts)
        node = bns.x[bns.changelist[i]]
        v = bns.bnd.bn.vertices[i].var
        arity = states(v)
        fac = bns.bnd.fg[node]
        pars = vars(fac) - v
        numparstates = nrStates(pars)
        for parstate in 1:numparstates, state in 1:arity
            index = conditionalState(v, pars, state, parstate)
            accum += log(fac[index]) * counts[i][index]
        end
        bns.fvalue[node] = -accum
    end
    return sum(bns.fvalue) - bns.logqfactor # TODO - or + logqfactor here?
end

function propose(bns::BayesNetSampler)
    bns.bnd.dirty = true
    clearBackups!(bns.bnd.fg)

    bns.logqfactor = 0.0
    numnodes = length(bns.bnd.bn.vertices)
    scheme = rand(1:3)

    if scheme==1 # Temporal order change
        k = rand(1:numnodes)
        #push!(get!(addlist, node, []), something)
        #push!(get!(addlist, node, []), something)
end

function reject(bns::BayesNetSampler)
    bns.bnd.dirty = true
    restoreFactors!(bns.bnd.fg)
end

function check_cpds(bnd::BayesNetDAI)
    for i=1:numFactors(bnd.fg)
        v = bnd.bn.vertices[i].var
        arity = states(v)
        parvars = vars(bnd.fg[i]) - v
        numparstates = nrStates(parvars)
        for parstate=1:numparstates
            tot = 0.0
            for state=1:arity
                index = conditionalState(v, parvars, state, parstate)
                tot += bnd.fg[i][index]
            end
            Base.Test.@test_approx_eq tot 1.0
        end
    end
end

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
    bnd.jt = JTree(bnd.fg)
    init!(bnd.jt)
    run!(bnd.jt)
    bnd.dirty = false
    accum = 0.0
    for i in 1:numFactors(bnd.fg)
        v = bnd.bn.vertices[i]
        arity = states(v.var)
        pars = vars(bnd.fg[i]) - v.var
        numparstates = nrStates(pars)
        margfac = marginal(bnd.jt, pars)
        for parstate in 1:numparstates
            accumsum = 0.0
            tot = 0.0
            for state in 1:arity
                temp = bnd.fg[i][conditionalState(v.var, pars, state, parstate)]
                accumsum -= temp == 0.0 ? 0.0 : temp * log2(temp)
                tot += temp
            end
            #Base.Test.@test_approx_eq tot 1.0
            accum += margfac[parstate] * accumsum
        end
    end
    bnd.memo_entropy = accum
    return accum
end

function naive_entropy(bnd::BayesNetDAI)
    numstates = 1
    accum = 0.0
    tot = 0.0
    allvars = VarSet([x.var for x=bnd.bn.vertices]...)
    for n in bnd.bn.vertices
        numstates *= states(n.var)
    end
    assert(numstates == nrStates(allvars))

    for i=1:numstates
        accumprod = 1.0
        statemap = calcState(allvars, i)
        for j in 1:numFactors(bnd.fg)
            reducedstate = calcLinearState(vars(bnd.fg[j]), statemap, allvars)
            accumprod *= bnd.fg[j][reducedstate]
        end
        tot += accumprod
        accum += accumprod == 0 ? 0.0 : accumprod * log2(accumprod)
    end
    assert((tot-1.0)<=eps())
    return -accum
end


#
#need to handle prior information: structure and cpd
#init cpds
#init structures

