import ocv.*;
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
ocvP5 ocv;
CascadeClassifier classifier;

ArrayList<Rect> faceRects;

void setup()
{
  // This is what you'll load if you're loading from MacPorts, otherwise this should be
  // wherever you built the OpenCV libraries
  System.load(new File("/opt/local/share/OpenCV/java/libopencv_java245.dylib").getAbsolutePath());

  ocv = new ocvP5(this);  
  size(640, 480);

  String[] cameras = Capture.list();
  cam = new Capture(this, cameras[0]);
  cam.start();
  
  classifier = new CascadeClassifier(dataPath("haarcascade_frontalface_default.xml"));
  
  faceRects = new ArrayList(); 
  stroke(255);
  noFill();
}

void draw() 
{

  if (cam.available() == true) 
  {
    cam.read();
    pimg = cam;
    Mat m = ocv.toCV(pimg);

    Mat gray = new Mat(m.rows(), m.cols(), CvType.CV_8U);
    Imgproc.cvtColor(m, gray, Imgproc.COLOR_BGRA2GRAY);

     MatOfRect objects = new MatOfRect();

    Size minSize = new Size(150, 150);
    Size maxSize = new Size(300, 300);

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


