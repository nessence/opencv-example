//
//  main.cpp
//  cvmagic
//
//  Created by Alex on 4/15/13.
//  Copyright (c) 2013 Alex Leverington. All rights reserved.
//

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Cocoa/Cocoa.h>
#endif

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;


// INSTRUCTIONS: Set image_name to path to an image and the subsequent sizes.


// Set Image Name
String image_name = "/PATH/TO/IMAGE";

// Set real-world size of poly (8.5x11 sheet of paper)
cv::Size landscape_object_size(110, 85);
cv::Size portrait_object_size(85, 110);

// Rotation
int thresh = 195;
int max_thresh = 255;
int rotate_x = 90;
int rotate_y = 90;
int rotate_z = 90; // 0.05

// Global (for the lazy example)
Mat src; Mat src_gray;

// Function header
void thresh_callback(int, void* );

void imshow_tosize(const string& winname, InputArray mat, cv::Size size) {
    Mat resized;
    resize(mat, resized, size, 0, 0, CV_INTER_AREA);
    imshow(winname, resized);
}

/** @function main */
int main( int argc, char** argv )
{
    // Load source image and convert it to gray
    Mat original = imread( image_name, 1 ); // argv[1]
    resize(original, src, original.size()); // drawContours crahses w/o this step
    
    // Convert image to gray
    cvtColor( src, src_gray, CV_BGR2GRAY );
    
    // manually call once for initial render
    thresh_callback( 0, 0 );
    
    // Show original w/controls
    const char* source_window = "Source";
    namedWindow( source_window, CV_WINDOW_NORMAL );
    imshow_tosize(source_window, src, cv::Size(src.cols/4,src.rows/4));
    
    createTrackbar( " Threshold:", "Source", &thresh, max_thresh, thresh_callback );
    createTrackbar( " Tilt X:", "Source", &rotate_x, 180, thresh_callback );
    createTrackbar( " Tilt Y:", "Source", &rotate_y, 180, thresh_callback );
    createTrackbar( " Rotate:", "Source", &rotate_z, 180, thresh_callback );
    
    waitKey(0);
    return(0);
}


void sortCorners(std::vector<cv::Point2f>& corners, cv::Point2f center)
{
    std::vector<cv::Point2f> top, bot;
    
    for (int i = 0; i < corners.size(); i++)
    {
        if (corners[i].y < center.y)
            top.push_back(corners[i]);
        else
            bot.push_back(corners[i]);
    }
    
    cv::Point2f tl = top[0].x > top[1].x ? top[1] : top[0];
    cv::Point2f tr = top[0].x > top[1].x ? top[0] : top[1];
    cv::Point2f bl = bot[0].x > bot[1].x ? bot[1] : bot[0];
    cv::Point2f br = bot[0].x > bot[1].x ? bot[0] : bot[1];
    
    corners.clear();
    corners.push_back(tl);
    corners.push_back(tr);
    corners.push_back(br);
    corners.push_back(bl);
}


cv::Point2f computeIntersect(cv::Vec4i a,
                             cv::Vec4i b)
{
	int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3], x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];

	if (float d = ((float)(x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))
	{
		cv::Point2f pt;
		pt.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d;
		pt.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d;
		return pt;
	}
	else
		return cv::Point2f(-1, -1);
}


/** @function thresh_callback */
void thresh_callback(int, void* )
{
    static vector<String>windows;
    for(int i = 0; i < windows.size(); i++)
        destroyWindow(windows[i]);
    destroyWindow("Contours");
    
    Mat threshold_output;
    vector<vector<cv::Point> > contours;
    vector<Vec4i> hierarchy;
    
    // Apply threshold and then find contours
    threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );
    findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
    
    
    // storage for contours
    vector<vector<cv::Point>> contours_poly( contours.size() );
    vector<cv::Rect> boundRect( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );
    vector<vector<cv::Point>> hull( contours.size() );
    
    // this should be based on aproximate min/max object size given resolution of camera image
    double minArea = 0.010*src.size().width*src.size().height;
    double maxArea = 15*minArea;
    
    // Approximate contours, hull, polygons + get bounding rects and circles
    for( int i = 0; i < contours.size(); i++ ) {
        // TODO: filter min/maxArea here instead of later
//        approxPolyDP( Mat(contours[i]), contours_poly[i], arcLength(Mat(contours[i]), true) * 0.02, true ); // ,,3|arcLength(Mat(contours[i]), true)*0.04,
        
        // moved approxPolyDP to use hull instead of contours
        convexHull(Mat(contours[i]), hull[i]);
        approxPolyDP( Mat(hull[i]), contours_poly[i], arcLength(Mat(contours[i]), true) * 0.02, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
    }
    
    
    // canvas for drawing output
    Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
    src.copyTo(drawing);

    int polys_shown_index = 0;
    for( int i = 0; i< contours.size(); i++ ) {
        
        // skip if area is too small/large, there aren't enough points for a square, or contour isn't convex
        double area = contourArea(hull[i]);
        if(!( area > minArea && area < maxArea && contours_poly[i].size() > 3 && hull[i].size() > 3 && isContourConvex( Mat(contours_poly[i]) ) )) {
            continue;
        }
        
        // dark green is contours_poly (seemingly worse for src of transform)
        // pink is hough lines
        // black is convexHull

        // draw countours_poly and minenclosingcircle (dark green)
//        drawContours( drawing, contours_poly, i, Scalar( 15, 127, 15 ), 5, CV_AA, noArray(), 0 );
        circle(drawing, center[i], radius[i], Scalar( 15, 127, 15 ));

        // draw convex hull in black to canvas
        //        drawContours(drawing, hull, i, Scalar(0,0,0));
        
        // TBR; it seems hull.size is always > 4 which is peculiar
        if(hull[i].size() > 4) {
            RotatedRect ellipse = fitEllipse(hull[i]);
            cv::ellipse(drawing, ellipse, Scalar(15,199,15)); // lgreen
        }

        // get minarearect of hull
        // most important value this will yield is size() of the poly
        // the size of the poly is useful to set parameters for houghlines
        RotatedRect rotatedRect = minAreaRect(hull[i]);
        Size2f hullSize = rotatedRect.size;
        cv::Rect hullBoundingRect = rotatedRect.boundingRect();
        rectangle(drawing, hullBoundingRect, Scalar(199,0,0)); // blue
        
        
        // contours of hull will be drawn to this canvas for detection by houghlines.
        // this is not useful at this time but will be useful as a simulation for
        // later when edge detection is limited to sides of poly (rather than findContours).
        //
        // hypothesis: Where this becomes useful is after convexHull is run. Without drawing and
        // redetecting the contours, running minAreaRect will result in clipping corners.
        // If houghlines is processed after drawing contours, we'll end up with straight
        // lines which will yield a more accurate (???) minAreaRect.
        Mat magic(drawing.size(), CV_8UC1, Scalar(0,0,0));

        // draw convex hull to binary canvas; important to use CV_AA so lines are more defined
        drawContours( magic, hull, i, Scalar(255,255,255), 1, CV_AA, vector<Vec4i>(), 0, cv::Point() );

        // find houghlines and skip if there aren't enough lines ot make a square
        // min width of hough line is 5% of 
        vector<Vec4i> hough_lines;
        double minLineLength = 0.25*min(hullSize.width, hullSize.height);
        HoughLinesP(magic, hough_lines, 1, CV_PI/180, 70, minLineLength, minLineLength*0.10); // was 70
        if(hough_lines.size() < 4){ cout << "insufficient houghLines: " << hough_lines.size(); continue; }

        double angle = 0.;
        for (unsigned i = 0; i < hough_lines.size(); ++i)
        {
            cv::line(drawing, cv::Point(hough_lines[i][0], hough_lines[i][1]),
                     cv::Point(hough_lines[i][2], hough_lines[i][3]), cv::Scalar(255, 127 ,255), 2);
            angle += atan2((double)hough_lines[i][3] - hough_lines[i][1],
                           (double)hough_lines[i][2] - hough_lines[i][0]);
        }
        angle /= hough_lines.size(); // mean angle, in radians.
        
        // find corners from hough lines by calculating intersections and ignoring
        // intersections from ~parallel lines which would exist
        std::vector<cv::Point2f> hough_intersections;
        float min_x = hullBoundingRect.x - 1;
        float min_y = hullBoundingRect.y - 1;
        float max_x = hullBoundingRect.x + hullBoundingRect.width + 1;
        float max_y = hullBoundingRect.y + hullBoundingRect.height + 1;
        for (int i = 0; i < hough_lines.size(); i++)
        {
            for (int j = i+1; j < hough_lines.size(); j++)
            {
                // only add if intersection is within bounds
                cv::Point2f pt = computeIntersect(hough_lines[i], hough_lines[j]);
                if (pt.x <= max_x && pt.x >= min_x && pt.y <= max_y && pt.y >= min_y) {
                    bool add_point = true;
                    
                    // crude (N*N) but fast, as there'll only be 4 points:
                    // don't add point if it's close to any existing point
                    for(int pti = 0; pti < hough_intersections.size(); pti++) {
                        Point2f existing = hough_intersections[pti];
                        if( abs(existing.x-pt.x) < 3 && abs(existing.y-pt.y) < 5 ) {
                            add_point = false;
                            break;
                        }
                    }
                    
                    if(add_point) {
                        hough_intersections.push_back(pt);
                        circle(drawing, pt, 3, Scalar(255,0,0), 0);
                    }
                    
                }
            }
        }

        std::vector<cv::Point2f> hough_corners = hough_intersections;
        
        // going to label based on whether houghlines were able to be used or not
        std::ostringstream stringStream;
        stringStream << "Quad ";
        
        // convert corners to float and calculate center
        // (center calculation could be abstracted away)
        std::vector<cv::Point2f> corners;
        Point2f center(0,0);
        if(hough_corners.size() == 4) {
            stringStream << "HoughLines: ";
            for (int k = 0; k < 4; k++) {
                Point2f point((float)hough_corners[k].x, (float)hough_corners[k].y);
                corners.push_back(point);
                center += point;
            }
        } else {
            stringStream << "contours_poly: ";
            for (int k = 0; k < contours_poly[i].size(); k++) {
                Point2f point((float)contours_poly[i][k].x, (float)contours_poly[i][k].y);
                corners.push_back(point);
                center += point;
            }
        }
        center *= (1. / corners.size());
        sortCorners(corners, center);
        
        // Draw corner points
        cv::circle(drawing, corners[0], 7, CV_RGB(255,0,0), 2);
        cv::circle(drawing, corners[1], 7, CV_RGB(0,255,0), 2);
        cv::circle(drawing, corners[2], 7, CV_RGB(0,0,255), 2);
        cv::circle(drawing, corners[3], 7, CV_RGB(255,255,255), 2);
        
        // Draw mass center
        cv::circle(drawing, center, 3, CV_RGB(255,255,0), 2);

        
        cv::Rect src_rect_estimate = boundingRect(corners);
        cv::Mat quad;
        if(src_rect_estimate.width > src_rect_estimate.height) {
            quad = cv::Mat::zeros(portrait_object_size, CV_8UC3);
        } else {
            quad = cv::Mat::zeros(landscape_object_size, CV_8UC3);
        }
        
        std::vector<cv::Point2f> quad_pts;
        quad_pts.push_back(cv::Point2f(0, 0));
        quad_pts.push_back(cv::Point2f(quad.cols, 0));
        quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));
        quad_pts.push_back(cv::Point2f(0, quad.rows));
        
        cv::Mat transmtx = cv::getPerspectiveTransform(corners, quad_pts);
        cv::warpPerspective(src, quad, transmtx, quad.size(), CV_INTER_CUBIC);

        
        // flatten and export transform
        int flat_nsz[2] = {1, 9};
        Mat exportmtx = transmtx.reshape(1, 2, flat_nsz); // flatten transform matrix; was 0,1

#ifdef __OBJC__
        NSMutableArray *valuelist = [NSMutableArray arrayWithCapacity:9*2];
        for(int i = 0; i < exportmtx.cols; i++) {
            [valuelist addObject:[NSNumber numberWithDouble:exportmtx.at<double>(0, i, 0)]];
        }
        
        NSDictionary *polyExport = @{@"transform":valuelist};
        NSLog(@"%@", polyExport);
#endif
        
//
//        // import and rebuild transform
//        Mat import(2, flat_nsz, CV_64F);
//        {
//            int i = 0;
//#ifdef __OBJC__
//            for(NSNumber *f in [polyExport objectForKey:@"transform"]) {
//                import.at<double>(0, i++, 0) = [f doubleValue];
//            }
//#endif
//        }
//        
//        int transmtx_nsz[2] = {3, 3};
//        import = import.reshape(1, 2, transmtx_nsz);

        
        stringStream << polys_shown_index++;
        
        // random rotation (for replacing create-cascade tool)
        double width = quad.size().width;
        double height = quad.size().height;

        // NOTE: Below is only for testing, as it requires focal length.
        // Rotation matrices around the X axis (values are counter-rotational)
        double dx, dy;
        dx = dy = -1.0;
        // alpha is x/tilt
        // beta is y
        // gamma is z (perspective)
        double alpha =  (rotate_x-90.0)*CV_PI/180; // maxxangle * (2.0 * rand() / RAND_MAX - 1.0);
        double beta = (rotate_y-90.0)*CV_PI/180; // ( maxyangle - fabs( alpha ) ) * (2.0 * rand() / RAND_MAX - 1.0);
        double gamma = (rotate_z-90.0)*CV_PI/180; // maxzangle * (2.0 * rand() / RAND_MAX - 1.0);

        double f = quad.size().width; // was 360
        double dz = f;
        
        // project 2d -> 3d
        Mat A1 = (Mat_<double>(4,3) <<
                  1, 0, -width/2,
                  0, 1, -height/2,
                  0, 0,    0,
                  0, 0,    1);
        
        // rotate around x axis
        Mat RX = (Mat_<double>(4, 4) <<
                 1,          0,           0, 0,
                 0, cos(alpha), -sin(alpha), 0,
                 0, sin(alpha),  cos(alpha), 0,
                 0,          0,           0, 1);
        
        // rotate around y axis
        Mat RY = (Mat_<double>(4, 4) <<
                  cos(beta), 0, -sin(beta), 0,
                  0, 1,          0, 0,
                  sin(beta), 0,  cos(beta), 0,
                  0, 0,          0, 1);
        
        // rotate around z axis
        Mat RZ = (Mat_<double>(4, 4) <<
                  cos(gamma), -sin(gamma), 0, 0,
                  sin(gamma),  cos(gamma), 0, 0,
                  0,          0,           1, 0,
                  0,          0,           0, 1);
        
        // Translation matrix on the Z axis (distance)
        double sx = 1.0, sy = 1.0;
        Mat T = (Mat_<double>(4, 4) <<
                 sx, 0, 0, dx,
                 0, sy, 0, dy,
                 0, 0, 1, dz, // was dist
                 0, 0, 0, 1);
        
        // Camera Intrisecs matrix 3D -> 2D
        // 1 should be replaced with focal
        Mat A2 = (Mat_<double>(3,4) <<
                  f, 0, width/2, 0,
                  0, f, height/2, 0,
                  0, 0,   1, 0);

        Mat R = RX*RY*RZ;
        Mat trans = A2 * (T* (R*A1) );

        warpPerspective(quad, quad, trans, quad.size(), CV_INTER_CUBIC);
        

        windows.push_back(stringStream.str());
        
        namedWindow(stringStream.str());
        moveWindow(stringStream.str(), src_gray.cols/4, 0.5*quad.size().height*(polys_shown_index-1));
        imshow(stringStream.str(), quad);
    }
    
    std::ostringstream threshTextString;
    threshTextString << "Threshold: " << thresh;
    putText(drawing, threshTextString.str(), cv::Point(10,24), FONT_HERSHEY_DUPLEX, 1.0, Scalar( 225, 225, 225 ));
    
    // Show in a window
    namedWindow( "Contours", CV_WINDOW_AUTOSIZE ); // was autosize
    imshow_tosize( "Contours", drawing, cv::Size(drawing.cols/6, drawing.rows/6) );
    moveWindow("Contours", 0, 300);
}

