function kontrol(net, image)

I = imread(image);
G = imresize(I, [224, 224]);

[Label, Prob] = classify(net,G);
imshow(G);
title({char(Label), num2str(max(Prob),2)});
end