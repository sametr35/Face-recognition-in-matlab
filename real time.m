web=webcam();
algilayici =vision.CascadeObjectDetector();


while true
        
    goruntu =snapshot(web);
    gri = rgb2gray(goruntu);
    bbox = step(algilayici,gri);
    if isempty(bbox)
       
        imshow(goruntu);
        title("görüntüde kimse yok");
        pause(1);
    else
        crop=imcrop(goruntu,(bbox(1:4)));
    
        resim = imresize(crop, [224, 224]);
        
        [Label, Prob] = classify(net,resim);
        isim=char(Label);
        deger=num2str(max(Prob));
        detpic=insertObjectAnnotation(im,"rectangle",bbox,isim+" "+deger);
        imshow(detpic);
        
    end
    
        
 end

