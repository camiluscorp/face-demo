package com.example.face;

import nu.pattern.OpenCV;

import java.util.List;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;


public class FaceDetector {
	 private final CascadeClassifier faceCascade;

	  /*public FaceDetector(String cascadePath) {
	    System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
	    
	    this.cascade = new CascadeClassifier(cascadePath);
	    if (cascade.empty()) throw new IllegalStateException("No se pudo cargar el cascade: " + cascadePath);
	  }*/
	 public FaceDetector(String cascadePath) {
		    //OpenCV.loadShared(); 
		    OpenCV.loadLocally();
		    this.faceCascade = new CascadeClassifier(cascadePath);
		    if (faceCascade.empty()) {
		      throw new IllegalStateException("No se pudo cargar el cascade: " + cascadePath);
		    }
		  }

	  public List<Rect> detectFaces(Mat bgr) {
	    Mat gray = new Mat();
	    Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY);
	    Imgproc.equalizeHist(gray, gray);
	    MatOfRect faces = new MatOfRect();
	    faceCascade.detectMultiScale(
	        gray, faces,
	        1.1,   // scaleFactor
	        5,     // minNeighbors
	        0,
	        new Size(80,80), new Size()
	    );
	    return faces.toList();
	  }
}
