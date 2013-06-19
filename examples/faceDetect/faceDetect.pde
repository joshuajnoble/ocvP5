import ocv.fun.*;
import processing.video.*;

import org.opencv.video.*;
import org.opencv.core.*;
import org.opencv.calib3d.*;
import org.opencv.contrib.*;
import org.opencv.objdetect.*;
import org.opencv.imgproc.*;
import org.opencv.utils.*;
import org.opencv.features2d.*;
import org.opencv.highgui.*;
import org.opencv.ml.*;
import org.opencv.photo.*;

import java.util.Vector;

PImage pimg;
Capture cam;
ocvP5 p;
CascadeClassifier classifier;

ArrayList<Rect> faceRects;

void setup()
{
  System.load(new File("/Users/joshua.noble/code/OPENCVgit/opencv/build/lib/libopencv_java245.dylib").getAbsolutePath());

  p = new ocvP5(this);  
  size(640, 480);

  String[] cameras = Capture.list();
  cam = new Capture(this, cameras[0]);
  cam.start();
  
  classifier = new CascadeClassifier(dataPath("haarcascade_frontalface_default.xml"));
  
  faceRects = new ArrayList();
}

void draw() 
{

  if (cam.available() == true) 
  {
    cam.read();
    pimg = cam;
    Mat m = p.toCV(pimg);

    Mat gray = new Mat(m.rows(), m.cols(), CvType.CV_8U);
    Imgproc.cvtColor(m, gray, Imgproc.COLOR_BGRA2GRAY);

     MatOfRect objects = new MatOfRect();

    Size minSize = new Size(150, 150);
    Size maxSize = new Size(300, 300);

//    classifier.detectMultiScale(m, objects, 1, 3, 
//    (cannyPruning ? CASCADE_DO_CANNY_PRUNING : 0) |
//      (findBiggestObject ? CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH : 0), 
//    minSize, maxSize);

    classifier.detectMultiScale(gray, objects, 1.1, 3, Objdetect.CASCADE_DO_CANNY_PRUNING | Objdetect.CASCADE_DO_ROUGH_SEARCH, minSize, maxSize);

    faceRects.clear();

    for (Rect rect: objects.toArray()) {
      faceRects.add(new Rect(rect.x, rect.y, rect.width, rect.height));
    }
  }

  image(cam, 0, 0);

  for (int i = 0; i < faceRects.size(); i++) {
    rect(faceRects.get(i).x, faceRects.get(i).y, faceRects.get(i).width, faceRects.get(i).height);
  }
}


