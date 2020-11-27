
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/affine.hpp>

using namespace cv;

//namespace MYSTEREORECT {


void mystereoRectify( InputArray _cameraMatrix1, InputArray _distCoeffs1,
                        InputArray _cameraMatrix2, InputArray _distCoeffs2,
                        Size imageSize, InputArray _Rmat, InputArray _Tmat,
                        OutputArray _Rmat1, OutputArray _Rmat2,
                        OutputArray _Pmat1, OutputArray _Pmat2,
                        OutputArray _Qmat, int flags,
                        double alpha, Size newImageSize,
                        Rect* validPixROI1, Rect* validPixROI2 );

void mycvStereoRectify( const CvMat* _cameraMatrix1, const CvMat* _cameraMatrix2,
                      const CvMat* _distCoeffs1, const CvMat* _distCoeffs2,
                      CvSize imageSize, const CvMat* matR, const CvMat* matT,
                      CvMat* _R1, CvMat* _R2, CvMat* _P1, CvMat* _P2,
                      CvMat* matQ, int flags, double alpha, CvSize newImgSize,
                      CvRect* roi1, CvRect* roi2 );

static void icvGetRectangles( const CvMat* cameraMatrix, const CvMat* distCoeffs,
                 const CvMat* R, const CvMat* newCameraMatrix, CvSize imgSize,
                 cv::Rect_<float>& inner, cv::Rect_<float>& outer );
//}
