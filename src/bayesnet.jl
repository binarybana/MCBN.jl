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
    bn::BayesNetDAI
    data::Matrix{Int}
    cpds
    template
end

#propose
#energy
#reject

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

