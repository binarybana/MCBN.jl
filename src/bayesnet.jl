export BayesNetDAI, BayesNetSampler

type BayesNetDAI
    verts::Vector{Var}
    fg::FactorGraph
    ia::InfAlg
    dirty::Bool
    memo_entropy::Float64
end

length(net::BayesNetDAI) = length(net.verts)

function BayesNetDAI(n::Int)
    verts = [Var(x,2) for x=1:n] # FIXME hardcoded binary
    fg = FactorGraph([Factor(x) for x in verts]...)
    ia = InfAlg(fg)
    BayesNetDAI(verts, fg, ia, true, -Inf)
end

function random_net(n::Int=5, weights=WeightVec([0.5,0.3,0.1]))
    count = -1
    cyclic = true
    local net
    while cyclic
        net = BayesNetDAI(n)
        if n==1
            move_params!(net,1)
            return net
        end
        for i=1:n
            for j=1:(sample(weights)-1)
                out = i
                while out == i
                    out = rand(1:n)
                end
                add_edge!(net, out, i)
            end
            move_params!(net,i)
        end
        g = net2graph(net)
        cyclic = test_cyclic_by_dfs(g)
        count += 1
    end
    #if count > 0
        #println("Generated $count cyclic graphs, restarting!")
    #end
    net
end

function net2graph(net)
    N = length(net.verts)
    g = simple_inclist(N)
    for i=1:N
        fac = net.fg[i]
        allvars = vars(fac)
        pars = allvars - net.verts[i]
        for n in labels(pars)
            add_edge!(g, n, i)
        end
    end
    g
end

function net2mat(net)
    g = net2graph(net)
    adjacency_matrix(g)
end

function draw_data(net, num)
    g = net2graph(net)
    tsort = topological_sort_by_dfs(g)
    data = zeros(Int, num_vertices(g), num)
    for i=1:num, j in tsort
        fac = net.fg[j]
        allvars = vars(fac)
        pars = allvars - net.verts[j]
        if length(pars) > 0
            pstates = Int[]
            for k in labels(pars)
                push!(pstates, data[findfirst(tsort,k),i])
            end
            pstate = calcLinearState(pars, pstates)
            i1,i2 = conditionalStateBoth(net.verts[j], pars, pstate)
            w = WeightVec([fac[i1], fac[i2]])
            data[j,i] = sample(1:2, w)
        else
            data[j,i] = sample(1:2, WeightVec(p(fac)))
        end
    end
    data
end

function clearBackups!(bnd::BayesNetDAI)
  clearBackups!(bnd.fg)
  bnd.dirty = true
end

function add_edge!(bnd::BayesNetDAI, u::Int, v::Int, backup=true)
    un,vn = bnd.verts[[u,v]]
    vnfac = bnd.fg[v]
    #assert(!(un in vars(vnfac)))
    un in vars(vnfac) && return
    bnd.dirty=true

    if backup
      setBackedFactor!(bnd.fg, v, embed(vnfac, un))
    else
      bnd.fg[v] = embed(vnfac, un) #Factor(vars(vnfac)+un)
    end
end

function del_edge!(bnd::BayesNetDAI, u::Int, v::Int, backup=true)
    un,vn = bnd.verts[[u,v]]
    bnd.dirty=true

    fac = bnd.fg[v]
    allvars = vars(fac)
    newvars = allvars - un
    newpars = newvars - vn
    newfac = Factor(newvars)

    for parstate in 1:nrStates(newpars) 
        # FIXME: hardcoded binary
        i1,i2 = conditionalState2(allvars, vn, un, newpars, 1, parstate)
        newval = (fac[i1] + fac[i2])/2
        c1,c2 = conditionalStateBoth(vn, newpars, parstate)
        newfac[c1] = newval
        newfac[c2] = 1 - newval
    end
    if backup
      setBackedFactor!(bnd.fg, v, newfac)
    else
      bnd.fg[v] = newfac
    end
end

function set_factor!(bnd::BayesNetDAI, i::Int, vals::Vector{Float64}, backup=true)
    bnd.dirty = true
    fcopy = bnd.fg[i]
    assert(length(vals) == nrStates(fcopy))
    oldvals = p(fcopy)
    oldvals[:] = vals
    if backup
      setBackedFactor!(bnd.fg, i, fcopy)
    else
      bnd.fg[i] = fcopy
    end
end

function move_params!(bnd::BayesNetDAI, node, backup=true)
    bnd.dirty = true

    fac = bnd.fg[node]
    curr = bnd.verts[node]
    pars = vars(fac) - curr

    for p in 1:nrStates(pars) # FIXME: hardcoded binary
        newval = rand()
        c1,c2 = conditionalStateBoth(curr, pars, p)
        fac[c1] = newval
        fac[c2] = 1 - newval
    end
    if backup
      setBackedFactor!(bnd.fg, node, fac)
    else
      bnd.fg[node] = fac
    end
    return 0.0 # FIXME hardcoded binary: new_count * lgamma(self.arity)
end

function adjust_factor!(bnd::BayesNetDAI, node, addlist, dellist, backup=true)
    for n in dellist
        del_edge!(bnd, n, node, backup)
    end
    for n in addlist
        add_edge!(bnd, n, node, backup)
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
    changelist::Vector{Int}
    x::Vector{Int} # node ordering/mapping
    oldx::Vector{Int}
    fvalue::Vector{Float64}
    oldfvalue::Vector{Float64}
    logqfactor::Float64
    mat::Matrix{Bool}
    oldmat::Matrix{Bool}
    limparent::Int
    prior::Function
end

function BayesNetSampler(net::BayesNetDAI, data::Matrix{Int}, prior=basic_prior::Function) 
    assert(all(0 .< data .<= 2)) # FIXME binary hardcode
    n = length(net)
    BayesNetSampler(net,
    data,
    [1:n;],
    randperm(n),
    randperm(n),
    zeros(Float64, n),
    zeros(Float64, n),
    0.0,
    zeros(Bool,n,n),
    zeros(Bool,n,n),
    4,
    prior)
end

function BayesNetSampler(n::Int, data::Matrix{Int}, prior=basic_prior::Function) 
    BayesNetSampler(BayesNetDAI(n), data, prior)
end

function show(io::IO, bns::BayesNetSampler)
    show(io, bns.mat[bns.x,bns.x])
end

function energy(bns::BayesNetSampler)

    eprior = bns.prior(bns)

    ##### energy from data (likelihood) ####
    counts = {}
    buf = Int[]
    for node in bns.changelist
      fac = bns.bnd.fg[node]
      facvs = vars(fac)
      inds = labels(fac)
      count = zeros(Int,nrStates(fac))
      for row=1:size(bns.data,2)
          for ind in inds
              push!(buf, bns.data[ind,row])
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
        fac = bns.bnd.fg[node]
        pars = vars(fac) - v
        numparstates = nrStates(pars)
        for parstate in 1:numparstates
            i1,i2 = conditionalStateBoth(v, pars, parstate) # Hardcoded binary
            accum += log(fac[i1]) * counts[i][i1]
            accum += log(fac[i2]) * counts[i][i2]
        end
        bns.fvalue[node] = -accum
    end
    return sum(bns.fvalue) - bns.logqfactor + eprior # TODO - or + logqfactor here?
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
        overparlimit = true
        local i, j, edgedel
        while overparlimit
            i = rand(1:numnodes)
            j = rand(1:numnodes)
            while i==j
                j = rand(1:numnodes)
            end
            if i > j
                i,j = j,i
            end

            edgedel = mat[i,j]
            mat[i,j] = !edgedel
            overparlimit = sum(mat[:,j]) > bns.limparent
            if overparlimit
                mat[i,j] = edgedel # reset for another loop
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

function record(bns::BayesNetSampler)
  return deepcopy(bns.bnd)
end

function check_bns(bns::BayesNetSampler)
    # Sanity check over topological node ordering
    n = size(bns.mat,1)
    for i=1:n, j=1:i
        assert(!bns.mat[i,j])
    end

    # Sanity checks over bayesnetdai
    check_bnd(bns.bnd)
    #@show bns.mat[bns.x,bns.x]
    #@show net2mat(bns.bnd)
    ord = sortperm(bns.x)
    Base.Test.@test all(bns.mat[ord,ord] .== net2mat(bns.bnd))
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

function controllability(bnd::BayesNetDAI, target_vars, target_states::Vector{Int}, control_var)
  # Measure the amount of shift in probabliity we obtain by fixing control_var
  # to either 0 or 1 in the prob. of target_vars being in target_states
  @assert typeof(target_states) == Vector{Int}
  clearBackups!(bnd)
  conbnd = deepcopy(bnd) # Don't want to mutate incoming bnd

  target_vars = DAI.VarSet(conbnd.verts[target_vars]...)
  cvar = conbnd.verts[control_var]
  cvarset = DAI.VarSet(cvar)

  tarstate = calcLinearState(target_vars, target_states)

  # @show conbnd.fg
  conbnd.fg[control_var] = Factor(cvar)

  set_factor!(conbnd, control_var, [0.0, 1.0], false)
  setupIA!(conbnd)
  clearBackups!(conbnd)
  # @show conbnd.fg
  jointfac = marginal(conbnd.ia, target_vars, true)
  # normalize!(jointfac)

  ssd1 = jointfac[tarstate]
  # @show "ssd1", jointfac

  set_factor!(conbnd, control_var, [1.0, 0.0], false)
  setupIA!(conbnd)
  clearBackups!(conbnd)
  jointfac = marginal(conbnd.ia, target_vars, true)
  # normalize!(jointfac)

  ssd2 = jointfac[tarstate]
  # @show "ssd2", jointfac

  setupIA!(bnd)
  clearBackups!(bnd)
  jointfac = marginal(bnd.ia, target_vars)

  origssd = jointfac[tarstate]
  # @show "origssd", jointfac

  # @show ssd1, ssd2, origssd
  return max(ssd1,ssd2) - origssd
end


function cod(bnd::BayesNetDAI, target::Int, predictors::Vector{Int})
  clearBackups!(bnd)
  setupIA!(bnd)
  targv = bnd.verts[target]
  targvs = VarSet(targv)
  predvs = VarSet(bnd.verts[predictors]...)

  margfac = marginal(bnd.ia, targvs)
  joint = marginal(bnd.ia, targvs+predvs)

  normalize!(margfac)
  normalize!(joint)

  numparstates = nrStates(predvs)
  epsdot = 0.0
  for parstate=1:numparstates
    i1,i2 = conditionalStateBoth(targv, predvs, parstate) # Hardcoded binary
    epsdot += min(joint[i1], joint[i2])
  end
  epszero = min(margfac[1], margfac[2])

  return 1.0 - epsdot/epszero
end

function setupIA!(bnd::BayesNetDAI)
  if bnd.dirty
    bnd.ia = InfAlg(bnd.fg)
    init!(bnd.ia)
    run!(bnd.ia)
  end
end

function kld(bnd::BayesNetDAI, other::BayesNetDAI)
    bnd.memo_entropy = entropy(bnd)
    # Now entropy and iaree are correct
    accum = 0.0
    for i=1:numFactors(bnd.fg)
      v = other.verts[i]
      arity = states(v)
      factor = other.fg[i]
      numparstates = div(nrStates(factor),arity)
      # Assumption: same ordering between bnd and other
      jointvars = vars(factor)
      parvars = jointvars - v
      joint = marginal(bnd.ia, jointvars)
      for parstate=1:numparstates
        i1,i2 = conditionalStateBoth(v, parvars, parstate) # Hardcoded binary
        accum += joint[i1] * log2(factor[i1])
        accum += joint[i2] * log2(factor[i2])
      end
    end
    return -bnd.memo_entropy - accum
end

function factor_klds(bnd1::MCBN.BayesNetDAI, bnd2::MCBN.BayesNetDAI, node::Int)
    bnd2.memo_entropy = MCBN.entropy(bnd2) # Prep the InfAlg
    accum = 0.0
    v = bnd1.verts[node]
    arity = states(v)
    factor = bnd1.fg[node]
    jointvars = vars(factor)
    parvars = jointvars - v
    numparstates = nrStates(parvars)
    joint = marginal(bnd2.ia, jointvars)
    for parstate=1:numparstates
      i1,i2 = conditionalStateBoth(v, parvars, parstate) # Hardcoded binary
      joint1 = joint[i1]
      joint2 = joint[i2]
      jointsum = joint1+joint2
      accum += factor[i1] * log2(factor[i1]/joint1*jointsum)
      accum += factor[i2] * log2(factor[i2]/joint2*jointsum)
    end
    return accum
end

function entropy(bnd::BayesNetDAI)
    if !bnd.dirty
        return bnd.memo_entropy
    end
    setupIA!(bnd)
    accum = 0.0
    for i in 1:numFactors(bnd.fg)
        v = bnd.verts[i]
        arity = states(v)
        pars = vars(bnd.fg[i]) - v
        numparstates = nrStates(pars)
        margfac = marginal(bnd.ia, pars)
        factor = bnd.fg[i]
        for parstate in 1:numparstates
            accumsum = 0.0
            i1,i2 = conditionalStateBoth(v, pars, parstate) # Hardcoded binary
            temp = factor[i1]
            accumsum -= temp == 0.0 ? 0.0 : temp * log2(temp)
            temp = factor[i2]
            accumsum -= temp == 0.0 ? 0.0 : temp * log2(temp)
            accum += margfac[parstate] * accumsum
        end
    end
    bnd.memo_entropy = accum
    bnd.dirty = false
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

function basic_prior(bns::BayesNetSampler)
    0.0
end

#need to handle prior information: structure and cpd
#init cpds
#init structures

