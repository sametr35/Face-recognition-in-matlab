web=webcam();
algilayici =vision.CascadeObjectDetector();


while true
        
    goruntu =snapshot(web);
    gri = rgb2gray(goruntu);
    bbox = step(algilayici,gri);
    
    
    resim = imresize(goruntu, [224, 224]);
        
    [Label, Prob] = classify(net,resim);
    isim=char(Label);
    deger=num2str(max(Prob));
    detpic=insertObjectAnnotation(goruntu,"rectangle",bbox,isim+" "+deger);
    imshow(detpic);
    
        
 end

