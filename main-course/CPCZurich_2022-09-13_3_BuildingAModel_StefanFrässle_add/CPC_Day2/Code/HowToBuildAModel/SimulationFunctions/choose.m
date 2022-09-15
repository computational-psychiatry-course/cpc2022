function a = choose(p)

% chose an action with respective probability
a = max(find([-eps cumsum(p)] < rand));

end
