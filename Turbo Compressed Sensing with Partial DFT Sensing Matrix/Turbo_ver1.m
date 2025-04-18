% For Proposed Turbo.
clear;
clc;

%% Params
N = 8192;
M = 5734;
lambda = 0.4;
SNR_db = 50;
iters = 50;

sigma2 = 10^(-SNR_db/10);

x = zeros(N,1);
select_x = (rand(N,1) < lambda);
sigma_x2 = 1 / lambda; 
x(select_x) = (randn(sum(select_x),1) + 1i*randn(sum(select_x),1)) * sqrt(sigma_x2/2);
z = FFT(x);
select_idx = sort(randperm(N, M));
y = z(select_idx) + sqrt(sigma2/2)*(randn(M,1) + 1i*randn(M,1));

%% Algorithm
z_pri_A = zeros(N,1);
v_pri_A = 1;
x_pri_A = IFFT(z_pri_A);
z_pri_B = FFT(x_pri_A);  % 初始化

v_pri_B = v_pri_A;
mse = zeros(iters,1);

for i = 1:iters
    % Module A: LMMSE
    factor = v_pri_A / (v_pri_A + sigma2);
    delta = zeros(N,1);
    delta(select_idx) = (y - z_pri_A(select_idx));
    z_post_A = z_pri_A + factor * delta;

    v_post_A = v_pri_A - (v_pri_A^2 / (v_pri_A + sigma2)) * M / N;
    x_post_A = IFFT(z_post_A);

    % 第一轮直接等于输入
    if i == 1
        x_ext_A = x_post_A;
        v_ext_A = v_post_A;
    else
        v_ext_A = 1 / max(eps, (1/v_post_A - 1/v_pri_A)); 
        x_ext_A = v_ext_A * (x_post_A / v_post_A - x_pri_A / v_pri_A);
    end
    z_ext_A = FFT(x_ext_A);

    % Module B
    x_pri_B = IFFT(z_ext_A);
    v_pri_B = v_ext_A;

    abs2 = abs(x_pri_B).^2;

    % likelihood
    likelihood_0 = 1/(pi*v_pri_B) * exp(-abs2 / v_pri_B);
    likelihood_1 = 1/(pi*(v_pri_B + sigma_x2)) * exp(-abs2 / (v_pri_B + sigma_x2));
    Pr = 1 ./ (1 + ((1 - lambda)/lambda) .* (likelihood_0 ./ likelihood_1));

    % 后验均值
    x_post_B = Pr .* ((sigma_x2 / (sigma_x2 + v_pri_B)) * ones(N,1)) .* x_pri_B;

    % 后验方差
    v_post_elem = Pr .* (sigma_x2 * v_pri_B / (sigma_x2 + v_pri_B)) + ...
                  Pr .* (1 - Pr) .* abs((sigma_x2 / (sigma_x2 + v_pri_B)) * x_pri_B).^2;
    v_post_B = mean(v_post_elem);

    % ext
    z_post_B = FFT(x_post_B);
    z_pri_B = FFT(x_pri_B);  

    v_ext_B = 1 / max(eps, (1/v_post_B - 1/v_pri_B));  
    z_ext_B = v_ext_B * (z_post_B / v_post_B - z_pri_B / v_pri_B);

    % 更新输入
    z_pri_A = z_ext_B;
    v_pri_A = v_ext_B;
    x_pri_A = IFFT(z_pri_A);

    % MSE
    mse(i) = mean(abs(x_post_B - x).^2);
    fprintf('Proposed Iter %2d: MSE = %.4e, v_pri_A = %.4e\n', i, mse(i), v_pri_A);
end

%% plot
figure;
semilogy(1:iters, mse, 'bo-','LineWidth',1.5);
xlabel('迭代次数');
ylabel('MSE');
title('Proposed Turbo Compressed Sensing MSE 收敛曲线');
grid on;

%% 单位正交FFT. IFFT.
function ret = FFT(ss)
    N = length(ss);
    ret = fft(ss)/sqrt(N);
end

function ret = IFFT(ss)
    N = length(ss);
    ret = ifft(ss)*sqrt(N);
end
