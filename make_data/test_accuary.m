grid = [10,20,30,40,50];
h = 1./grid;
e = zeros(size(grid));
for i=1:size(grid, 2)
    e(i) = compute_error(grid(i));
end
p = polyfit(log(h), log(e), 1);
figure()
plot(log(h),log(e),'-bs','MarkerSize',5,'LineWidth',2)
title("fitting curve")
text(log(h(3))+0.2, log(e(3)),sprintf("log(e) = %.2f*log(h)%.2f",p(1),p(2)))
print(gcf,'-dpdf','test_accuary')

%compute_error(3)