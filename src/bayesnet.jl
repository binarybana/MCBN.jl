export BayesNetDAI, BayesNetSampler

type BayesNetDAI
    verts::Vector{Var}
    fg::FactorGraph
    jt::JTree
    dirty::Bool
    memo_entropy::Float64
end

function BayesNetDAI(n::Int)
    g = BayesNet(n)
    BayesNetDAI(g)
end

function BayesNetDAI(n::Int)
    verts = [Var(x,2) for x=1:n] # FIXME hardcoded binary
    fg = FactorGraph([Factor(x) for x in verts]...)
    jt = JTree(fg)
    BayesNetDAI(verts, fg, jt, true, -Inf)
end

function add_edge!(bnd::BayesNetDAI, u::Int, v::Int)
    un,vn = bnd.verts[[u,v]]
    vnfac = bnd.fg[v]
    assert(!(un in vars(vnfac)))
    bnd.dirty=true
    bnd.fg[v] = embed(vnfac, vars(vnfac)+un)
end

function del_edge!(bnd::BayesNetDAI, u::Int, v::Int)
    un,vn = bnd.verts[[u,v]]
    bnd.dirty=true

    fac = bnd.fg[v]
    newvars = vars(fac) - un
    newpars = newvars - vn
    newfac = Factor(newvars)

    for parstate in 1:nrStates(newpars), vs in 1:2 # FIXME: hardcoded binary
        newval = (fac[conditionalState2(vn, un, newpars, vs, 1, parstate)] +
            fac[conditionalState2(vn, un, newpars, vs, 2, parstate)])/2
        newfac[conditionalState(vn, newpars, 1, parstate)] = newval
        newfac[conditionalState(vn, newpars, 2, parstate)] = 1 - newval
    end
    bnd.fg[v] = newfac
    #bnd.fg[v] = marginal(vnfac, vars(vnfac)-un) # IIRC, this doesn't properly
    #handle the CPD structure, but it would otherwise replace most the above
    #code
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
    curr = bnd.verts[node]
    pars = vars(fac) - curr

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
        del_edge!(bnd, n, node)
    end
    for n in addlist
        add_edge!(bnd, n, node)
    end
    return 0.0 # FIXME hardcoded binary: new_count * lgamma(self.arity)
end

function show(io::IO, bnd::BayesNetDAI)
    show(io, bnd.verts)
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

type BayesNetSampler <: Sampler
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
    randperm(n),
    randperm(n),
    zeros(Float64, n),
    zeros(Float64, n),
    0.0,
    eye(Bool, n),
    eye(Bool, n))
end

function show(io::IO, bns::BayesNetSampler)
    show(io, bns.mat[bns.x,bns.x])
end

function factor_klds(bnd1::BayesNetDAI, bnd2::BayesNetDAI)
    0 #FIXME
end

function energy(bns::BayesNetSampler)
    #limit number of parents
    if any(sum(bns.mat, 2) .> bns.limparent + 1) # +1 -> diagonal # FIXME: this is actually pretty slow
        return 1e20
    end

    # energy contribution from structural
    adj_diff = sum(abs(bns.mat[bns.x,bns.x] - bns.template)) 
    e_struct = bns.p_structural * adj_diff

    # energy contribution from cpds
    cpd_diff = factor_klds(bns.bnd, bns.cpds)
    e_cpd = bns.p_cpds * cpd_diff

    ##### energy from data (likelihood) ####
    counts = {}
    buf = Int[]
    for node in bns.changelist
        fac = bns.bnd.fg[node]
        facvs = vars(fac)
        inds = labels(fac)
        count = zeros(Int,nrStates(fac))
        for row=1:size(bns.data,1)
            for ind in inds
                push!(buf, bns.data[row,ind])
            end
            index = calcLinearState(facvs, buf)
            empty!(buf)
            count[index] += 1
        end
        push!(counts, count)
    end

    for i=1:length(counts) # Calculate likelihood
        accum = 0.0
        node = bns.changelist[i]
        v = bns.bnd.verts[node]
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
    return sum(bns.fvalue) - bns.logqfactor + e_cpd + e_struct # TODO - or + logqfactor here?
end

function propose!(bns::BayesNetSampler)
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
    numnodes = length(bns.bnd.verts)
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
            underparlimit = sum(mat[:,j]) > bns.limparent+1 #+1 for diagonal
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

function reject!(bns::BayesNetSampler)
    bns.bnd.dirty = true
    bns.x[:] = bns.oldx
    bns.fvalue[:] = bns.oldfvalue
    bns.mat[:] = bns.oldmat
    restoreFactors!(bns.bnd.fg)
end

function save!(bns::BayesNetSampler)
    bns.oldx[:] = bns.x
    bns.oldfvalue[:] = bns.fvalue
    bns.oldmat[:] = bns.mat
    clearBackups!(bns.bnd.fg)
end

function check_bns(bns::BayesNetSampler)
    # Sanity check over topological node ordering
    n = size(bns.mat,1)
    for i=1:n, j=1:i
        if i==j
            assert(bns.mat[i,j])
        else
            assert(!bns.mat[i,j])
        end
    end

    # Sanity checks over bayesnetdai
    check_bnd(bns.bnd)
end

function check_bnd(bnd::BayesNetDAI)
    for i=1:numFactors(bnd.fg)
        v = bnd.verts[i]
        Base.Test.@test v in bnd.fg[i]
        arity = states(v)
        parvars = vars(bnd.fg[i]) - v
        numparstates = nrStates(parvars)
        for parstate=1:numparstates
            tot = 0.0
            for state=1:arity
                index = conditionalState(v, parvars, state, parstate)
                amt = bnd.fg[i][index]
                Base.Test.@test 0.0 <= amt <= 1.0
                tot += amt
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
        v = other.verts[i]
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
        v = bnd.verts[i]
        arity = states(v)
        pars = vars(bnd.fg[i]) - v
        numparstates = nrStates(pars)
        margfac = marginal(bnd.jt, pars)
        for parstate in 1:numparstates
            accumsum = 0.0
            tot = 0.0
            for state in 1:arity
                temp = bnd.fg[i][conditionalState(v, pars, state, parstate)]
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
    allvars = VarSet(bnd.verts...)
    for n in bnd.verts
        numstates *= states(n)
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

