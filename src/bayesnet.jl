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

function del_edge!(bn::BayesNet, un::Node, vn::Node)
    u = vertex_index(un)
    v = vertex_index(vn)
    ind = findfirst(bn.adjlist[u], v)
    @show u, ind
    @show bn.adjlist[u]
    deleteat!(bn.adjlist[u], ind)
    bn.nedges -= 1
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

function make_vertex(g::BayesNet, x) 
    ind = num_vertices(g)+1
    v = Var(ind,2) #FIXME hardcoded binary
    Node(ind, string(x), v) 
end
vertex_index(n::Node) = n.index

function adjmat(bn::BayesNet)
    n = length(bn.vertices)
    mat = zeros(Bool,n,n)
    for i in 1:n, neigbor=bn.adjlist[i]
        mat[i,vertex_index(neighbor)] = true
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

function del_edge!(bnd::BayesNetDAI, u::Int, v::Int)
    un,vn = bnd.bn.vertices[[u,v]]
    vnfac = bnd.fg[v]
    bnd.dirty=true
    bnd.fg[v] = marginal(vnfac, vars(vnfac)-un.var)
    del_edge!(bnd.bn, un, vn)
end

function set_factor!(bnd::BayesNetDAI, i::Int, vals::Vector{Float64})
    bnd.dirty = true
    fcopy = bnd.fg[i]
    assert(length(vals) == nrStates(fcopy))
    oldvals = p(fcopy)
    oldvals[:] = vals
    bnd.fg[i] = fcopy
end

function move_params!(bnd::BayesNetDAI, node)
    bnd.dirty = true

    fac = bnd.fg[node]
    pars = vars(fac)
    curr = Var(node,2) # FIXME: hardcoded binary
    erase!(pars, curr)

    for p in 1:nrStates(pars) # FIXME: hardcoded binary
        newval = rand()
        fac[conditionalState(curr, pars, 1, p)] = newval
        fac[conditionalState(curr, pars, 2, p)] = 1 - newval
    end
    bnd.fg[node] = fac
    return 0.0 # FIXME hardcoded binary: new_count * lgamma(self.arity)
end

function adjust_factor!(bnd::BayesNetDAI, node, addlist, dellist)
    for n in dellist
        del_edge!(bnd, node, n)
    end
    for n in addlist
        add_edge!(bnd, node, n)
    end
    #bnd.dirty = true
    #fac = bnd.fg[node]
    #fvars = vars(fac)
    #for n in dellist
        #erase!(fvars, Var(n,2)) # FIXME: hardcoded binary
    #end
    #fac = marginal(fac, fvars)
    #for n in addlist
        #insert!(fvars, Var(n,2)) # FIXME: hardcoded binary
    #end
    #fac = embed(fac, fvars)

    #bnd.fg[node] = fac
    check_bnd(bnd) # I suspect this will fail
    return 0.0 # FIXME hardcoded binary: new_count * lgamma(self.arity)
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
    cpds::BayesNetDAI
    p_structural::Float64
    template::Matrix{Bool}
    limparent::Int
    changelist::Vector{Int}
    x::Vector{Int} # node ordering/mapping
    oldx::Vector{Int}
    fvalue::Vector{Float64}
    oldfvalue::Vector{Float64}
    logqfactor::Float64
    mat::Matrix{Bool}
    oldmat::Matrix{Bool}
end

function BayesNetSampler(n::Int, data::Matrix{Int}) 
    assert(all(0 .< data .<= 2)) # FIXME binary hardcode
    
    BayesNetSampler(
    BayesNetDAI(n),
    data,
    0.2,
    BayesNetDAI(n),
    0.2,
    eye(Bool, n),
    4,
    [1:n],
    shuffle([1:n]),
    [1:n],
    zeros(Float64, n),
    zeros(Float64, n),
    0.0,
    eye(Bool, n),
    eye(Bool, n))
end

function factor_klds(bnd1::BayesNetDAI, bnd2::BayesNetDAI)
    0 #FIXME
end

function energy(bns::BayesNetSampler)
    # FIXME Untested code below!
    #
    # energy contribution from structural
    adj_diff = sum(abs(bns.mat[bns.x,bns.x] - bns.template)) 
    e_struct = bns.p_structural * adj_diff

    # energy contribution from cpds
    cpd_diff = factor_klds(bns.bnd, bns.cpds)
    e_cpd = bns.p_cpds * cpd_diff

    ##### energy from data (likelihood) ####
    accum = 0.0

    counts = {}
    for node in bns.changelist
        #limit number of parents
        if length(bns.bnd.bn.adjlist[node]) > bns.limparent
            return 1e20
        end
        push!(counts, zeros(Int,nrStates(bns.bnd.fg[node])))
    end

    allvars = VarSet([x.var for x=bns.bnd.bn.vertices]...)
    for i=1:length(counts) # Assemble count data structure
        node = bns.changelist[i]
        fac = bns.bnd.fg[node]
        facvs = vars(fac)
        inds = labels(fac)
        for row=1:size(bns.data,1)
            index = calcLinearState(facvs, bns.data[inds]) 
            counts[node][index] += 1
        end
    end

    for i=1:length(counts) # Calculate likelihood
        node = bns.changelist[i]
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
    bns.oldx[:] = bns.x
    bns.oldfvalue[:] = bns.fvalue
    bns.oldmat[:] = bns.mat

    bns.logqfactor = 0.0
    resize!(bns.changelist, 0)

    bns.logqfactor = 0
    addlist = (Int=>Vector{Int})[]
    dellist = (Int=>Vector{Int})[]
    x = bns.x # convenience labels
    mat = bns.mat 
    numnodes = length(bns.bnd.bn.vertices)
    scheme = rand(1:3)
    if scheme==1 # Temporal order change
        k = rand(1:numnodes-1)

        #For any parents of node k not shared with k+1 (and vice versa):
        for i=1:k-1
            if mat[i,k] && !mat[i,k+1]
                push!(get!(dellist, x[k], []), x[i])
                push!(get!(addlist, x[k+1], []), x[i])
            elseif !mat[i,k] && mat[i,k+1]
                push!(get!(dellist, x[k+1], []), x[i])
                push!(get!(addlist, x[k], []), x[i])
            end
        end
        push!(bns.changelist, x[k], x[k+1])

        #For any children of node k not shared with k+1 (and vice versa):
        for j=k+2:numnodes
            if mat[k,j] && !mat[k+1,j]
                push!(get!(dellist, x[j], []), x[k])
                push!(get!(addlist, x[j], []), x[k+1])
                push!(bns.changelist, x[j])
            elseif !mat[k,j] && mat[k+1,j]
                push!(get!(dellist, x[j], []), x[k+1])
                push!(get!(addlist, x[j], []), x[k])
                push!(bns.changelist, x[j])
            end
        end

        if mat[k,k+1]
            push!(get!(dellist, x[k+1], []), x[k])
            push!(get!(addlist, x[k], []), x[k+1])
        end

        bns.x[k], bns.x[k+1] = bns.x[k+1], bns.x[k]

    elseif scheme==2 # skeletal change
        underparlimit = true
        local i, j, edgedel
        while underparlimit
            i = rand(1:numnodes)
            j = rand(1:numnodes)
            while i==j
                j = rand(1:numnodes)
            end
            if i > j
                i,j = j,i
            end

            edgedel = mat[i,j]
            mat[i,j] = !mat[i,j]
            underparlimit = sum(mat[:,j]) > bns.limparent
            if underparlimit
                mat[i,j] = !mat[i,j] # reset for another loop
            end
        end
        push!(bns.changelist, x[j])

        if edgedel
            push!(get!(dellist, x[j], []), x[i])
        else
            push!(get!(addlist, x[j], []), x[i])
        end

    elseif scheme==3 # Null (parameter only) move

        k = rand(1:numnodes)

        bns.logqfactor = move_params!(bns.bnd, bns.x[k])
        push!(bns.changelist, x[k])
    end

    if scheme != 3
        for node in union(Set(keys(addlist)), Set(keys(dellist)))
            bns.logqfactor += adjust_factor!(bns.bnd, node, 
                get(addlist, node,[]), 
                get(dellist, node,[]))
        end
    end

    return scheme
end

function reject(bns::BayesNetSampler)
    bns.bnd.dirty = true
    bns.x[:] = bns.oldx
    bns.fvalue[:] = bns.oldfvalue
    bns.mat[:] = bns.oldmat
    restoreFactors!(bns.bnd.fg)
end

function check_bns(bns::BayesNetSampler)
    # Sanity check over topological node ordering
    # TODO

    # Sanity checks over bayesnetdai
    check_bnd(bns.bnd)
end

function check_bnd(bnd::BayesNetDAI)
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

