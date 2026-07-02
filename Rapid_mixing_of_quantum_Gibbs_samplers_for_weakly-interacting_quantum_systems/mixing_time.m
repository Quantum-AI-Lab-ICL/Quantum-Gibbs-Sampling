function [] = mixing_time()

    %%%%%%%%%% inverse temperature %%%%%%%%%
    beta = 1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




    %%%%%%%%%%%%%%%% Hamiltonian %%%%%%%%%%%%%%%%%%%%
    
    %%% 1D spinful Fermi-Hubbard model
    t = 1;
    U = 1;  
    l = 3;
    n = 2*l;
    H = OneD_spinful_Fermi_Hubbard_Hamiltonian(t,U,l,n);
    

    % %%% 1D spinless Fermi-Hubbard model
    % t = 1;
    % U = 1;  
    % n = 6;
    % H = OneD_spinless_Fermi_Hubbard_Hamiltonian(t,U,n);


    % %%% 1D transverse-field Ising model
    % J = 0.5;
    % h = 1;
    % n = 6;
    % H = OneD_TFIM_Hamiltonian(n,J,h);


    % %%% 2D transverse-field Ising model
    % J = 0.5;
    % h = 1;
    % l = 2;
    % m = 3;
    % n = l*m;
    % H = TwoD_TFIM_Hamiltonian(l,m,J,h);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




    %%%%%%%%%%%%%%%    initial state    %%%%%%%%%%%%%%%%
    %%% the maximally mixed state
    rho = 1/(2^n) * eye(2^n); 

    % %%% the all zero state
    % rho = zeros(2^n,2^n);
    % rho(1,1) = 1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




    %%%%%%%%%% distance epsilon from the Gibbs state in trace norm %%%%%%%%
    tolerance = 0.01;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    mixing_time = LindbladianMixingTime(beta,H,n,tolerance,rho, jumps = "Majoranas", filter = "Gaussian", alsoEvaluateGap = false);


end


function [result] = FilterFunction(i,t,v,beta,filter)
%%%%% i and t represent the site and type of jump; but these options are not used here, and the filter function is taken to always be the same one

    if(filter == "Gaussian")
        result = exp(-beta^2*v^2 /8 - beta*v/4);
    elseif(filter == "Metropolis")
        result = exp(-(beta*v + sqrt(1+beta^2 * v^2))/4) * exp(-20/(100 - min(10,abs(v))^2));
    end

end


function [c] = createOperator(i,n)

    XGate = sparse([0 1; 1 0]);
    YGate = sparse([0 -1i; 1i 0]);
    ZGate = sparse([1 0 ; 0 -1]);

    c = speye(2^n,2^n);

    XiGate = kron(kron(speye(2^(i-1)),XGate),speye(2^(n-i)));
    YiGate = kron(kron(speye(2^(i-1)),YGate),speye(2^(n-i)));

    for j = 1:i-1

        ZjGate = kron(kron(speye(2^(j-1)),ZGate),speye(2^(n-j)));

        c = - c * ZjGate;

    end

    c = c * (XiGate + 1j * YiGate)/2;


end

function [c] = annihilateOperator(i,n)

    XGate = sparse([0 1; 1 0]);
    YGate = sparse([0 -1i; 1i 0]);
    ZGate = sparse([1 0 ; 0 -1]);

    c = speye(2^n,2^n);

    XiGate = kron(kron(speye(2^(i-1)),XGate),speye(2^(n-i)));
    YiGate = kron(kron(speye(2^(i-1)),YGate),speye(2^(n-i)));

    for j = 1:i-1

        ZjGate = kron(kron(speye(2^(j-1)),ZGate),speye(2^(n-j)));

        c = - c * ZjGate;

    end

    c = c * (XiGate - 1j * YiGate)/2;


end





function [M] = singleParticleHamiltonian(t,n)
%%%% written in canonical fermions

    M = zeros(n,n);

    for i = 1:n-1

        M(i,i+1) = -t;
        M(i+1,i) = -t;

    end
        
end


function [H] = OneD_spinless_Fermi_Hubbard_Hamiltonian(t,U,n)

    M = singleParticleHamiltonian(t,n);

    H = sparse(2^n,2^n);

    for i = 1:n

        for j = 1:n

            if(M(i,j) ~= 0)

                H = H + M(i,j)*createOperator(i,n)*annihilateOperator(j,n);

            end

        end

    end

    for i = 1:n-1

        H = H + U * createOperator(i,n) * annihilateOperator(i,n) * createOperator(i+1,n) * annihilateOperator(i+1,n);

    end


end


function [H] = OneD_spinful_Fermi_Hubbard_Hamiltonian(t,U,l,n)

    M = singleParticleHamiltonian(t,l);

    H = sparse(2^n,2^n);


    %%%% spin up
    for i = 1:l

        for j = 1:l

            if(M(i,j) ~= 0)

                H = H + M(i,j)*createOperator(i,n)*annihilateOperator(j,n);

            end

        end

    end

    %%%% spin down
    for i = 1:l

        for j = 1:l

            if(M(i,j) ~= 0)

                H = H + M(i,j)*createOperator(i+l,n)*annihilateOperator(j+l,n);

            end

        end

    end



    for i = 1:l

        H = H + U * createOperator(i,n) * annihilateOperator(i,n) * createOperator(i+l,n) * annihilateOperator(i+l,n);

    end


end



function [H] = OneD_TFIM_Hamiltonian(n,J,h)

    H = sparse(2^n,2^n);

    XGate = sparse([0 1; 1 0]);
    ZGate = sparse([1 0 ; 0 -1]);

    for i = 1:n-1

        ZiGate = kron(kron(speye(2^(i-1)),ZGate),speye(2^(n-i)));
        ZjGate = kron(kron(speye(2^(i)),ZGate),speye(2^(n-i-1)));

        H = H - J * ZiGate * ZjGate;


    end


    for i = 1:n

        XiGate = kron(kron(speye(2^(i-1)),XGate),speye(2^(n-i)));

        H = H - h * XiGate;


    end

end


function [H] = TwoD_TFIM_Hamiltonian(l,m,J,h)

    %%%% width l, height m
   
    n = l*m;

    H = sparse(2^n,2^n);

    XGate = sparse([0 1; 1 0]);
    ZGate = sparse([1 0 ; 0 -1]);


    for j=1:m

        for i=1:l-1

            index1 = (j-1)*l + i;
            index2 = (j-1)*l + i + 1;

            ZiGate = kron(kron(speye(2^(index1-1)),ZGate),speye(2^(n-index1)));
            ZjGate = kron(kron(speye(2^(index2-1)),ZGate),speye(2^(n-index2)));
    
            H = H - J * ZiGate * ZjGate;

        end

    end


    for i=1:l

        for j=1:m-1

            index1 = (j-1)*l + i;
            index2 = j*l + i;

            ZiGate = kron(kron(speye(2^(index1-1)),ZGate),speye(2^(n-index1)));
            ZjGate = kron(kron(speye(2^(index2-1)),ZGate),speye(2^(n-index2)));
    
            H = H - J * ZiGate * ZjGate;

        end

    end



   for i = 1:n

        XiGate = kron(kron(speye(2^(i-1)),XGate),speye(2^(n-i)));

        H = H - h * XiGate;


   end

end






function [A] = JumpOperators_Majoranas(i,t,n)
%%%% single site Majoranas, i specifies the site, t the type

    if(t == 1)
        A = createOperator(i,n) + annihilateOperator(i,n);
    else
        A = -1j * (annihilateOperator(i,n) - createOperator(i,n));
    end

end



function [A] = JumpOperators_Paulis(i,t,n)
%%%% single site Pauli gates, i specifies the site, t the type

    XGate = [0 1; 1 0];
    YGate = [0 -1i;1i 0];
    ZGate = [1 0 ; 0 -1];

    if(t == 1)

        Gate = XGate;

    elseif(t == 2)

        Gate = YGate;

    else

        Gate = ZGate;

    end

    A = kron(kron(speye(2^(i-1)),sparse(Gate)),speye(2^(n-i)));
end



function [L] = LindbladOperatorsEfficient(i,t,n,V,D,beta,jumps,filter)
    
    if(jumps == "Paulis")
        A = JumpOperators_Paulis(i,t,n);
    elseif(jumps == "Majoranas")
        A = JumpOperators_Majoranas(i,t,n);
    end


    vector = zeros(2^(2*n),1);

    %%%%%%% first multiplying

    for i = 1:2^n

        s = transpose(A(i,:));

        a = transpose(conj(V(i,:)));

        c = kron(a,transpose(transpose(s)*V));

        vector = vector + c;

    end

    %%%%%%%%% second multiplying

    for i = 1:2^n
                
        eig1 = D(i);

        for j = 1:2^n

            eig2 = D(j);

            vector((i-1)*2^n+j) = vector((i-1)*2^n+j) * FilterFunction(i,t,eig1-eig2,beta,filter);

        end

    end

    newvector = zeros(2^(2*n),1);

    %%%%%%% third multiplying

    for i = 1:2^n

        a = V(:,i);

        B = conj(V);

        s = vector((i-1)*2^n+1 : i*2^n);

        c = kron(a,B*s);

        newvector = newvector + c;

    end

    L = zeros(2^n,2^n);

    for i = 1:2^n

        for j = 1:2^n

            L(i,j) = newvector((i-1)*2^n+j);

        end

    end

end


function [mixingtime] = LindbladianMixingTime(beta,H,n,tolerance,rho,options)

    arguments

        beta double
        H
        n {mustBeInteger, mustBePositive}
        tolerance double
        rho
        options.jumps string {mustBeMember(options.jumps,["Paulis","Majoranas"])} = "Paulis"
        options.filter string {mustBeMember(options.filter,["Gaussian","Metropolis"])} = "Gaussian"
        options.timestep double = 0.1/(n * log(n))
        options.order {mustBeInteger, mustBePositive} = 4
        options.alsoEvaluateGap logical = false
        options.verbosity logical = true

    end

    timestep = options.timestep;
    order = options.order;
    verbosity = options.verbosity;
    jumps = options.jumps;
    filter = options.filter;

    if(jumps == "Majoranas")
        type_max = 2;
    elseif(jumps == "Paulis")
        type_max = 3;
    end


    if(verbosity)
        disp("Diagonalising H")
    end

    [V, D] = eig(full(H));

    D = diag(D);

    Z = 0;

    GSEnergy = min(D);

    for i = 1:2^n

        Z = Z + exp(-beta*(D(i)-GSEnergy));

    end

    GibbsD = zeros(1,2^n);

    for i = 1:2^n

        GibbsD(i) = exp(-beta*(D(i)-GSEnergy))/Z;

    end

    GibbsMatrix = V * diag(GibbsD) * ctranspose(V);


    %%%%%%%%% preparing the Lindbladian

    LindOperators = cell(n,type_max);

    for i = 1:n
        if(verbosity)
            disp("Preparing Lindbladian action "+num2str(i)+"/"+num2str(n))
        end

        for t = 1:type_max
            if(verbosity)
                disp(num2str(t)+"/"+num2str(type_max))
            end

            LindOperators(i,t) = {LindbladOperatorsEfficient(i,t,n,V,D,beta,jumps,filter)};

        end

    end



    S = zeros(2^n,2^n);

    for i = 1:n

        for t = 1:type_max

            L = cell2mat(LindOperators(i,t));

            S = S + 1/2 * ctranspose(L) * L;

        end

    end



    if(verbosity)
        disp("Starting coherent term")
    end

    vector = zeros(2^(2*n),1);

    %%%%%%% first multiplying

    for i = 1:2^n

        s = transpose(S(i,:));

        a = transpose(conj(V(i,:)));

        c = kron(a,transpose(transpose(s)*V));

        vector = vector + c;

    end

    %%%%%%%%% second multiplying

    for i = 1:2^n

        for j = 1:2^n

            vector((i-1)*2^n+j) = vector((i-1)*2^n+j) * (-1j * tanh(-beta/4*(D(i)-D(j))));  

        end

    end

    newvector = zeros(2^(2*n),1);

    %%%%%%% third multiplying

    for i = 1:2^n

        a = V(:,i);

        B = conj(V);

        s = vector((i-1)*2^n+1 : i*2^n);

        c = kron(a,B*s);

        newvector = newvector + c;

    end

    G = zeros(2^n,2^n);

    for i = 1:2^n

        for j = 1:2^n

            G(i,j) = newvector((i-1)*2^n+j);

        end

    end




    function [finalvector] = evaluateAction(vector)

        sigma = zeros(2^n,2^n);

        for newi = 1:2^n

            for newj = 1:2^n
    
                sigma(newi,newj) = vector((newi-1)*2^n+newj);
    
            end

        end

        M = (- 1j * G - S) * sigma + sigma * (1j * G - S);
        
        for newi = 1:n
    
            for newt = 1:type_max
    
                newL = cell2mat(LindOperators(newi,newt));
    
                M = M + newL * sigma * ctranspose(newL);
    
            end
    
        end
  
        finalvector = zeros(2^(2*n),1);

        for newi = 1:2^n

            for newj = 1:2^n
    
                finalvector((newi-1)*2^n+newj) = M(newi,newj);
    
            end

        end

    end





    GibbsVector = zeros(2^(2*n),1);

    for i = 1:2^n

        for j = 1:2^n

            GibbsVector((i-1)*2^n+j) = GibbsMatrix(i,j);

        end

    end


    if(verbosity)
        check = max(abs(evaluateAction(GibbsVector)),[],"all") %%%%% should be 0
    end





    
%%%%%% calculating the evolved state using small time steps and series expansions at each step

    condition = 10;
    mixingtime = 0;


    rho = reshape(transpose(rho),[2^(2*n),1]);
    Evolution = rho;

    while condition > tolerance

        mixingtime = mixingtime + timestep;


        sum = Evolution;
        newpower = Evolution;

        for power = 1:order

            newpower = evaluateAction(newpower);

            sum = sum + (timestep^power)/factorial(power) * newpower;

        end

        Evolution = sum;

        EvolvedMatrix = transpose(reshape(Evolution,[2^n,2^n]));

        difference = EvolvedMatrix - GibbsMatrix;

        %%% evaluating the trace distance from the Gibbs state
        condition = real(trace(sqrtm(ctranspose(difference)*difference)));
   
    end

    disp("Mixing time was "+num2str(mixingtime))






%%%%%% evaluation of the spectral gap

    if(options.alsoEvaluateGap)
    
        if(verbosity)
            disp("Starting diagonalisation")
        end
    
    
        shiftvalue = eigs(@(sigma) evaluateAction(sigma),2^(2*n),1,"largestabs","IsFunctionSymmetric",false);
    
        if(verbosity)
            disp("First shift done, lowest eigenvalue of the Lindbladian was " + num2str(shiftvalue))
        end

        norm_here = ctranspose(GibbsVector) * GibbsVector;

    end
    
    function [finalvector] = ShiftedEigenvalues(vector)

        finalvector = evaluateAction(vector);

        prod = 1/norm_here * ctranspose(GibbsVector) * vector;

        finalvector = finalvector - shiftvalue * vector + shiftvalue/2 * prod * GibbsVector;

    end
    
    if(options.alsoEvaluateGap)

    
        gap = - eigs(@(sigma) ShiftedEigenvalues(sigma),2^(2*n),1,"largestabs","IsFunctionSymmetric",false) - shiftvalue;
    
    
        if (gap < 0 || isnan(gap))
    
            if(verbosity)
                disp("Error, gap was found to be negative or nan - happens due to rounding errors when the actual gap is of a similar magnitude as a rounding error")
                disp("Need to use implicitly restarted Arnoldi")
            end

            evalues = eigs(@(sigma) evaluateAction(sigma),2^(2*n),2,"largestreal","SubspaceDimension",min(500,2^(2*n)),"IsFunctionSymmetric",false);
    
            gap = evalues(1)-evalues(2);
    
        end
    
        disp("Spectral gap between highest eigenvalues was "+num2str(gap))

    end



end