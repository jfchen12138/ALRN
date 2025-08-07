function err = compute_error(N)
    %N = 50;  
    h = 1/N; %% grid size
    dx = 0 : h : 1;
    dy = 0 : h : 1;
    dz = 0 : h : 1;
	%size(dx)
    [X,Y,Z] = meshgrid(dx,dy,dz);
    u_true = sin(pi*X).*sin(pi*Y).*sin(pi*Z);
    f = 3 * pi^2 * sin(pi*X).*sin(pi*Y).*sin(pi*Z);
    
    
    f = reshape(permute(f(2:end-1, 2:end-1, 2:end-1),[3,2,1]), [], 1);
    %f = reshape(f(2:end-1, 2:end-1, 2:end-1),[],1);
    u_pred = zeros(N+1, N+1, N+1);
    coef = ones(N+1, N+1, N+1);  %%% (x, y, z)
    %size(coef)
    coef_int = coef(2:end-1, 2:end-1, 2:end-1);
    %size(coef_int)
    coef_int_1d = reshape(permute(coef_int, [3,2,1]),[],1);
    coef_pad = padarray(coef, [1, 1, 1], 1, 'both');
    %size(coef_pad)
    coef_grad_x = (coef_pad(3:end, 2:end-1, 2:end-1) - coef_pad(1:end-2, 2:end-1, 2:end-1))/2/h;
    coef_grad_y = (coef_pad(2:end-1, 3:end, 2:end-1) - coef_pad(2:end-1, 1:end-2, 2:end-1))/2/h;
    coef_grad_z = (coef_pad(2:end-1, 2:end-1, 3:end) - coef_pad(2:end-1, 2:end-1, 1:end-2))/2/h;
    % size(coef_grad_x)
    % size(coef_grad_y)
    % size(coef_grad_z)
    coef_grad_x_int_1d = reshape(permute(coef_grad_x(2:end-1, 2:end-1, 2:end-1), [3,2,1]), [], 1);
    coef_grad_y_int = coef_grad_y(2:end-1, 2:end-1, 2:end-1);
    coef_grad_z_int = coef_grad_z(2:end-1, 2:end-1, 2:end-1);
   % A matrix for u_x
    A = sparse(1:(N-2)*(N-1)^2,(N-1)^2+1:(N-1)^3, coef_grad_x_int_1d(1:(N-2)*(N-1)^2)/2/h,(N-1)^3, (N-1)^3)...
        + sparse((N-1)^2+1:(N-1)^3, 1:(N-2)*(N-1)^2, -coef_grad_x_int_1d((N-1)^2+1: end)/2/h,(N-1)^3, (N-1)^3);
    %size(A)
    %A
    %full(A)
    % B matrix for u_y
    B = sparse((N-1)^3, (N-1)^3);
    for j=1:N-1
        k = (j-1)*(N-1)^2;
        m = reshape(permute(coef_grad_y_int(j, :, :), [3,2,1]),[], 1)/2/h;
        B = B +  sparse(k+1:k+(N-2)*(N-1), k+N: k+(N-1)^2, m(1:(N-2)*(N-1)), (N-1)^3,(N-1)^3)+...
            sparse(k+N:k+(N-1)^2, k+1:k+(N-2)*(N-1), -m(N:(N-1)^2), (N-1)^3, (N-1)^3);
    end
    %full(B)
    % c matrix for u_z
    C = sparse((N-1)^3, (N-1)^3);
    for j=1:N-1
        for k=1:N-1
            ind = (j-1)*(N-1)^2+(k-1)*(N-1);
            m = reshape(permute(coef_grad_z_int(j,k,:), [3,2,1]),[],1)/2/h;
            %size(m)
            C = C + sparse(ind+1:ind+N-2, ind+2:ind+N-1, m(1:end-1), (N-1)^3, (N-1)^3)...
                + sparse(ind+2: ind+N-1, ind+1:ind+N-2, -m(2:end), (N-1)^3, (N-1)^3);
        end
    end
    %full(C)
            
    %D matrix for u_xx
    D =sparse(1:(N-2)*(N-1)^2,(N-1)^2+1:(N-1)^3, coef_int_1d(1:(N-2)*(N-1)^2)/h/h,(N-1)^3, (N-1)^3)...
        + sparse((N-1)^2+1:(N-1)^3, 1:(N-2)*(N-1)^2, coef_int_1d((N-1)^2+1: end)/h/h,(N-1)^3, (N-1)^3)...
        + sparse(1:(N-1)^3, 1:(N-1)^3, -2 * coef_int_1d/h/h, (N-1)^3, (N-1)^3);
    %full(D)
    % E matrix for u_yy
    E = sparse((N-1)^3, (N-1)^3);
    for j=1:N-1
        k = (j-1)*(N-1)^2;
        m = reshape(permute(coef_int(j, :, :), [3,2,1]),[], 1)/h/h;
        E = E +  sparse(k+1:k+(N-2)*(N-1), k+N: k+(N-1)^2, m(1:(N-2)*(N-1)), (N-1)^3,(N-1)^3)+...
            sparse(k+N:k+(N-1)^2, k+1:k+(N-2)*(N-1), m(N:(N-1)^2), (N-1)^3, (N-1)^3)+...
            sparse(k+1:k+(N-1)^2, k+1:k+(N-1)^2, -2* m, (N-1)^3, (N-1)^3);
    end
    %full(E)
    %F matrix for u_zz
    F = sparse((N-1)^3, (N-1)^3);
    for j=1:N-1
        for k=1:N-1
            ind = (j-1)*(N-1)^2+(k-1)*(N-1);
            m = reshape(permute(coef_int(j,k,:), [3,2,1]),[],1)/h/h;
            F = F + sparse(ind+1:ind+N-2, ind+2:ind+N-1, m(1:end-1), (N-1)^3, (N-1)^3)...
                + sparse(ind+2: ind+N-1, ind+1:ind+N-2, m(2:end), (N-1)^3, (N-1)^3)...
                + sparse(ind+1: ind+N-1, ind+1:ind+N-1, -2*m, (N-1)^3, (N-1)^3);
        end
    end
    %full(F)
    
    M = A+B+C+D+E+F;
    x = -M\f;
    u_pred(2:end-1, 2:end-1, 2:end-1) = permute(reshape(x, N-1, N-1,N-1), [3,2,1]);
%     u_true(:,:,end-1)
%     u_pred(:,:,end-1)
    err = max(abs(u_true - u_pred),[], 'all');