
N = 20;
Nsamples = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7];
es = zeros(length(Nsamples), N);

for n = 1:N

    for i = 1:length(Nsamples)
        rr = rand([Nsamples(i), 4]);
        rr(:,1) = rr(:,1) * 2 * pi;
        rr(:,2) = rr(:,2) * pi;
        rr(:,3) = rr(:,3) * 2 * pi;
        rr(:,4) = rr(:,4) * pi;

        I = sum(-tricky_integral(rr(:,1) , rr(:,2), rr(:,3), rr(:,4)))/Nsamples(i);


        D = (2*pi)^2 * pi^2;


        es(i,n) = D * I;

    end

end
figure(1); clf; subplot(2,1,1); hold on; grid on;
semilogx(Nsamples, es, 'b.-', 'LineWidth', 0.1);
%plot(Nsamples, mean(es(end,:))*ones(size(Nsamples)), 'r', 'LineWidth', 0.1);
set(gca, 'XScale', 'log');
ylabel('Quantity of Interest Value');
title('MC Convergence Scheme')

subplot(2,1,2); hold on; grid on;
semilogx(Nsamples, abs(es(end) - es)/es(end)*100);
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
xlabel('Number of Monte Carlo Evaluation Points');
ylabel('Percent Convergence')

fprintf("MC solution = %.4f +/- %.4f (%.3f %%)\n", ...
    mean(es(end,:)), std(es(end,:)), std(es(end,:))/mean(es(end,:)) * 100);

function f = tricky_integral(phi_d, th_d, phi_s, th_s)

f = sin(th_s).^4 .* sin(th_d) .* (cos(th_d) + sin(th_d)) .* ( sin(th_s) .* (sin(th_d) + cos(th_d)) < 0 );

end
