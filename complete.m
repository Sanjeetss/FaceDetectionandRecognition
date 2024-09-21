% % face detection
web=webcam('FINGERS 1080 Hi-Res');
im=snapshot(web);
load myNet;
dete=vision.CascadeObjectDetector();

while true
    im=snapshot(web);
    im2=rgb2gray(im);
    bbox=step(dete,im2);
   if(sum(sum(bbox))~=0)
        es=imcrop(im,bbox(1,:));
        es=imresize(es,[227 227]);
        label=classify(myNet,es);
        drawnow;
        pic=insertObjectAnnotation(im,'rectangle',bbox,char(label));
        imshow(pic);
    else
        title('No Face Detected');
        imshow(pic);
    end
end

