tic
data_num = 1024;
epsilon = 0.1;
N = 40;
a = zeros(data_num, N+1, N+1, N+1);
u = zeros(data_num, N+1, N+1, N+1);
for i=1:data_num
    sprintf("%d", i)
    h = 1/N; %% grid size
    dx = 0 : h : 1;
    dy = 0 : h : 1;
    dz = 0 : h : 1;
    coef = 1 - epsilon*rand(N+1, N+1, N+1);  %%% (x, y, z)
    a(i,:,:,:) = coef;
    u_pred = zeros(N+1, N+1, N+1);
    f = ones((N-1)^3,1);
    coef_int = coef(2:end-1, 2:end-1, 2:end-1);
    coef_int_1d = reshape(permute(coef_int, [3,2,1]),[],1);
    coef_pad = padarray(coef, [1, 1, 1], 1, 'both');
    coef_grad_x = (coef_pad(3:end, 2:end-1, 2:end-1) - coef_pad(1:end-2, 2:end-1, 2:end-1))/2/h;
    coef_grad_y = (coef_pad(2:end-1, 3:end, 2:end-1) - coef_pad(2:end-1, 1:end-2, 2:end-1))/2/h;
    coef_grad_z = (coef_pad(2:end-1, 2:end-1, 3:end) - coef_pad(2:end-1, 2:end-1, 1:end-2))/2/h;
    coef_grad_x_int_1d = reshape(permute(coef_grad_x(2:end-1, 2:end-1, 2:end-1), [3,2,1]), [], 1);
    coef_grad_y_int = coef_grad_y(2:end-1, 2:end-1, 2:end-1);
    coef_grad_z_int = coef_grad_z(2:end-1, 2:end-1, 2:end-1);
   % A matrix for u_x
    A = sparse(1:(N-2)*(N-1)^2,(N-1)^2+1:(N-1)^3, coef_grad_x_int_1d(1:(N-2)*(N-1)^2)/2/h,(N-1)^3, (N-1)^3)...
        + sparse((N-1)^2+1:(N-1)^3, 1:(N-2)*(N-1)^2, -coef_grad_x_int_1d((N-1)^2+1: end)/2/h,(N-1)^3, (N-1)^3);
    % B matrix for u_y
    B = sparse((N-1)^3, (N-1)^3);
    for j=1:N-1
        k = (j-1)*(N-1)^2;
        m = reshape(permute(coef_grad_y_int(j, :, :), [3,2,1]),[], 1)/2/h;
        B = B +  sparse(k+1:k+(N-2)*(N-1), k+N: k+(N-1)^2, m(1:(N-2)*(N-1)), (N-1)^3,(N-1)^3)+...
            sparse(k+N:k+(N-1)^2, k+1:k+(N-2)*(N-1), -m(N:(N-1)^2), (N-1)^3, (N-1)^3);
    end
    % c matrix for u_z
    C = sparse((N-1)^3, (N-1)^3);
    for j=1:N-1
        for k=1:N-1
            ind = (j-1)*(N-1)^2+(k-1)*(N-1);
            m = reshape(permute(coef_grad_z_int(j,k,:), [3,2,1]),[],1)/2/h;
            C = C + sparse(ind+1:ind+N-2, ind+2:ind+N-1, m(1:end-1), (N-1)^3, (N-1)^3)...
                + sparse(ind+2: ind+N-1, ind+1:ind+N-2, -m(2:end), (N-1)^3, (N-1)^3);
        end
    end
            
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
    u(i,:,:,:) = u_pred;
end
size(a)
%save("/home/jfchen/galerkin-transformer/data/data_darcy_3d_.mat", 'a', 'u','-v7.3')
save("/home/jfchen/Lrk/data/darcy3d_r41_N1024_valid.mat", 'a', 'u')
toc