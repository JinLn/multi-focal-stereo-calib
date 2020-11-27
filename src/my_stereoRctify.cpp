   
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/affine.hpp>
#include "include/my_stereoRctify.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;
//using namespace MYSTEREORECT;
    // cameraMatrix1-第一个摄像机的摄像机矩阵
    // distCoeffs1-第一个摄像机的畸变向量
    // cameraMatrix2-第二个摄像机的摄像机矩阵
    // distCoeffs1-第二个摄像机的畸变向量
    // imageSize-图像大小
    // R- stereoCalibrate() 求得的R矩阵
    // T- stereoCalibrate() 求得的T矩阵
    // R1-输出矩阵，第一个摄像机的校正变换矩阵（旋转变换）
    // R2-输出矩阵，第二个摄像机的校正变换矩阵（旋转矩阵）
    // P1-输出矩阵，第一个摄像机在新坐标系下的投影矩阵
    // P2-输出矩阵，第二个摄像机在想坐标系下的投影矩阵
    // Q-4*4的深度差异映射矩阵
    // flags-可选的标志有两种零或者 CV_CALIB_ZERO_DISPARITY ,如果设置 CV_CALIB_ZERO_DISPARITY 的话，该函数会让两幅校正后的图像的主点有相同的像素坐标。否则该函数会水平或垂直的移动图像，以使得其有用的范围最大
    // alpha-拉伸参数。如果设置为负或忽略，将不进行拉伸。如果设置为0，那么校正后图像只有有效的部分会被显示（没有黑色的部分），如果设置为1，那么就会显示整个图像。设置为0~1之间的某个值，其效果也居于两者之间。
    // newImageSize-校正后的图像分辨率，默认为原分辨率大小。
    // validPixROI1-可选的输出参数，Rect型数据。其内部的所有像素都有效
    // validPixROI2-可选的输出参数，Rect型数据。其内部的所有像素都有效

void mystereoRectify( InputArray _cameraMatrix1, InputArray _distCoeffs1,
                        InputArray _cameraMatrix2, InputArray _distCoeffs2,
                        Size imageSize, InputArray _Rmat, InputArray _Tmat,
                        OutputArray _Rmat1, OutputArray _Rmat2,
                        OutputArray _Pmat1, OutputArray _Pmat2,
                        OutputArray _Qmat, int flags,
                        double alpha, Size newImageSize,
                        Rect* validPixROI1, Rect* validPixROI2 )
{
    Mat cameraMatrix1 = _cameraMatrix1.getMat(), cameraMatrix2 = _cameraMatrix2.getMat();
    Mat distCoeffs1 = _distCoeffs1.getMat(), distCoeffs2 = _distCoeffs2.getMat();
    Mat Rmat = _Rmat.getMat(), Tmat = _Tmat.getMat();
    CvMat c_cameraMatrix1 = cameraMatrix1;
    CvMat c_cameraMatrix2 = cameraMatrix2;
    CvMat c_distCoeffs1 = distCoeffs1;
    CvMat c_distCoeffs2 = distCoeffs2;
    CvMat c_R = Rmat, c_T = Tmat;

    int rtype = CV_64F;
    _Rmat1.create(3, 3, rtype);
    _Rmat2.create(3, 3, rtype);
    _Pmat1.create(3, 4, rtype);
    _Pmat2.create(3, 4, rtype);
    CvMat c_R1 = _Rmat1.getMat(), c_R2 = _Rmat2.getMat(), c_P1 = _Pmat1.getMat(), c_P2 = _Pmat2.getMat();
    CvMat c_Q, *p_Q = 0;

    if( _Qmat.needed() )
    {
        _Qmat.create(4, 4, rtype);
        p_Q = &(c_Q = _Qmat.getMat());
    }

    CvMat *p_distCoeffs1 = distCoeffs1.empty() ? NULL : &c_distCoeffs1;
    CvMat *p_distCoeffs2 = distCoeffs2.empty() ? NULL : &c_distCoeffs2;
    mycvStereoRectify( &c_cameraMatrix1, &c_cameraMatrix2, p_distCoeffs1, p_distCoeffs2,
        imageSize, &c_R, &c_T, &c_R1, &c_R2, &c_P1, &c_P2, p_Q, flags, alpha,
        newImageSize, (CvRect*)validPixROI1, (CvRect*)validPixROI2);
}



void mycvStereoRectify( const CvMat* _cameraMatrix1, const CvMat* _cameraMatrix2,
                      const CvMat* _distCoeffs1, const CvMat* _distCoeffs2,
                      CvSize imageSize, const CvMat* matR, const CvMat* matT,
                      CvMat* _R1, CvMat* _R2, CvMat* _P1, CvMat* _P2,
                      CvMat* matQ, int flags, double alpha, CvSize newImgSize,
                      CvRect* roi1, CvRect* roi2 )
{
    double _om[3], _t[3] = {0}, _uu[3]={0,0,0}, _r_r[3][3], _pp[3][4];
    double _ww[3], _wr[3][3], _z[3] = {0,0,0}, _ri[3][3], _w3[3];
    cv::Rect_<float> inner1, inner2, outer1, outer2;

    CvMat om  = cvMat(3, 1, CV_64F, _om);
    CvMat t   = cvMat(3, 1, CV_64F, _t);
    CvMat uu  = cvMat(3, 1, CV_64F, _uu);
    CvMat r_r = cvMat(3, 3, CV_64F, _r_r);
    CvMat pp  = cvMat(3, 4, CV_64F, _pp);
    CvMat ww  = cvMat(3, 1, CV_64F, _ww); // temps
    CvMat w3  = cvMat(3, 1, CV_64F, _w3); // temps
    CvMat wR  = cvMat(3, 3, CV_64F, _wr);
    CvMat Z   = cvMat(3, 1, CV_64F, _z);
    CvMat Ri  = cvMat(3, 3, CV_64F, _ri);
    double nx = imageSize.width, ny = imageSize.height;
    int i, k;
    double nt, nw;
    // 罗德里格斯 src为输入的旋转向量（3x1或者1x3）或者旋转矩阵（3x3） dst为输出的旋转矩阵（3x3）或者旋转向量（3x1或者1x3）
    if( matR->rows == 3 && matR->cols == 3 )
        cvRodrigues2(matR, &om);          // get vector rotation
    else
        // cvConvert函数用于图像和矩阵之间的相互转换 用cvConvert 把IplImage转为矩阵
        cvConvert(matR, &om); // it's already a rotation vector 
    // 函数 cvConvertScale 有多个不同的目的因此就有多个同义函数（如上面的#define所示）。
    // 该函数首先对输入数组的元素进行比例缩放，然后将shift加到比例缩放后得到的各元素上，
    // 即： dst(I)=src(I)*scale + (shift,shift,…)，最后可选的类型转换将结果拷贝到输出数组。
    cvConvertScale(&om, &om, -0.5); // get average rotation
    cvRodrigues2(&om, &r_r);        // rotate cameras to same orientation by averaging
    // cvMatMul(src1,src2,dst);　表示dst=src1*src2 即
    cvMatMul(&r_r, matT, &t);
    // 求 浮点数x的绝对值
    int idx = fabs(_t[0]) > fabs(_t[1]) ? 0 : 1;

    // if idx == 0
    //   e1 = T / ||T||
    //   e2 = e1 x [0,0,1]

    // if idx == 1
    //   e2 = T / ||T||
    //   e1 = e2 x [0,0,1]

    // e3 = e1 x e2

    _uu[2] = 1;
    // cvCrossProduct(&Va, &Vb, &Vc) 向量积: Va x Vb -> Vc end{verbatim} 注意Va, Vb, Vc 在向量积中向量元素个数须相同. 
    cvCrossProduct(&uu, &t, &ww);
    //　计算数组的绝对范数， 绝对差分范数或者相对差分范数
    // arr1    第一输入图像
    // arr2    第二输入图像 ，如果为空（NULL）, 计算 arr1 的绝对范数，否则计算 arr1-arr2 的绝对范数或者相对范数。 normType  范数类型，
    // norm = ||arr1||C = maxI abs(arr1(I)), 如果 normType = CV_C 
    // norm = ||arr1||L1 = sumI abs(arr1(I)), 如果 normType = CV_L1 
    // norm = ||arr1||L2 = sqrt( sumI arr1(I)2), 如果 normType = CV_L2 
    //https://blog.csdn.net/u013935644/article/details/53397921?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522160644202619726891112493%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=160644202619726891112493&biz_id
    // =0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-53397921.pc_first_rank_v2_rank_v28&utm_term=cvNorm&spm=1018.2118.3001.4449
    nt = cvNorm(&t, 0, CV_L2);
    nw = cvNorm(&ww, 0, CV_L2);
    cvConvertScale(&ww, &ww, 1 / nw);
    cvCrossProduct(&t, &ww, &w3);
    nw = cvNorm(&w3, 0, CV_L2);
    cvConvertScale(&w3, &w3, 1 / nw);
    _uu[2] = 0;

    for (i = 0; i < 3; ++i)
    {
        _wr[idx][i] = -_t[i] / nt;
        _wr[idx ^ 1][i] = -_ww[i];
        _wr[2][i] = _w3[i] * (1 - 2 * idx); // if idx == 1 -> opposite direction
    }

    // apply to both views
    //  double cvGEMM(//矩阵的广义乘法运算
    // 	const CvArr* src1,//乘数矩阵
    // 	const CvArr* src2,//乘数矩阵
    // 	double alpha,//1号矩阵系数
    // 	const CvArr* src3,//加权矩阵
    // 	double beta,//2号矩阵系数
    // 	CvArr* dst,//结果矩阵
    // 	int tABC = 0//变换标记
    // );
    // CV_GEMM_A_T 转置 src1
    // CV_GEMM_B_T 转置 src2
    // CV_GEMM_C_T 转置 src3
    // dst = (alpha*src1)xsrc2+(beta*src3)
    cvGEMM(&wR, &r_r, 1, 0, 0, &Ri, CV_GEMM_B_T);
    cvConvert( &Ri, _R1 );
    cvGEMM(&wR, &r_r, 1, 0, 0, &Ri, 0);
    cvConvert( &Ri, _R2 );
    cvMatMul(&Ri, matT, &t);

    // calculate projection/camera matrices
    // these contain the relevant rectified image internal params (fx, fy=fx, cx, cy)
    double fc_new = DBL_MAX;
    CvPoint2D64f cc_new[2] = {{0,0}, {0,0}};

    newImgSize = newImgSize.width * newImgSize.height != 0 ? newImgSize : imageSize;//h还是原尺寸imageSize
    const double ratio_x = (double)newImgSize.width / imageSize.width / 2; //＝0.5
    const double ratio_y = (double)newImgSize.height / imageSize.height / 2; //＝0.5
    // idx=1 ===> ratio = ratio_x = 0.5
    const double ratio = idx == 1 ? ratio_x : ratio_y;
    //int idx = fabs(_t[0]) > fabs(_t[1]) ? 0 : 1;
    // 在访问CvMat数据时，比如用cvmGet 和 cvmSet ，矩阵的索引是从0、0开始的
    //    cvmGet(_cameraMatrix1, idx ^ 1, idx ^ 1) = fy
    //fc_new = (cvmGet(_cameraMatrix1, idx ^ 1, idx ^ 1) + cvmGet(_cameraMatrix2, idx ^ 1, idx ^ 1)) * ratio;
    fc_new = cvmGet(_cameraMatrix1, idx ^ 1, idx ^ 1);
    cout <<"fc_new =" << fc_new <<"\tcvmGet(_cameraMatrix1, idx ^ 1, idx ^ 1)="<<cvmGet(_cameraMatrix1, idx ^ 1, idx ^ 1)<<
           "\tcvmGet(_cameraMatrix2, idx ^ 1, idx ^ 1)="<<cvmGet(_cameraMatrix2, idx ^ 1, idx ^ 1)<<"\tratio="<<ratio<<endl;
    // cc_new[0].x=c1x , cc_new[0].y =c1y
    // cc_new[1].x=c2x , cc_new[1].y =c2y

    for( k = 0; k < 2; k++ )
    {
        const CvMat* A = k == 0 ? _cameraMatrix1 : _cameraMatrix2;
        const CvMat* Dk = k == 0 ? _distCoeffs1 : _distCoeffs2;
        CvPoint2D32f _pts[4];
        CvPoint3D32f _pts_3[4];
        CvMat pts = cvMat(1, 4, CV_32FC2, _pts);
        CvMat pts_3 = cvMat(1, 4, CV_32FC3, _pts_3);

        for( i = 0; i < 4; i++ )
        {
            int j = (i<2) ? 0 : 1;
            _pts[i].x = (float)((i % 2)*(nx));
            _pts[i].y = (float)(j*(ny));
        }
        // cvUndistortPoints函数校正的是理想点（ideal points，也就是[R|t]M'），要得到真正的图像点必须乘以摄像机内参。
        // 但是如果对最后一个参数进行赋值，也就是说给定摄像机校正后的内参，得到的结果也是图像点。
        cvUndistortPoints( &pts, &pts, A, Dk, 0, 0 );
        //转换为齐次坐标  
        cvConvertPointsHomogeneous( &pts, &pts_3 );

        //Change camera matrix to have cc=[0,0] and fc = fc_new
        double _a_tmp[3][3];
        CvMat A_tmp  = cvMat(3, 3, CV_64F, _a_tmp);
        _a_tmp[0][0]=fc_new;
        _a_tmp[1][1]=fc_new;
        _a_tmp[0][2]=0.0;
        _a_tmp[1][2]=0.0;
//        void cvProjectPoints2 //计算三维点在平面中的坐标
//        (
//        const CvMat* objectPoints, //是需要投影的点的序列，是一个点位置的N*3的矩阵。
//        const CvMat* rvec,
//        const CvMat* tvec, //建立两个坐标系的联系
//        const CvMat* cameraMatrix,
//        const CvMat* distCoeffs,//内参数矩阵和形变系数
//        CvMat* imagePoints,//N*2的矩阵将被写入计算结果
//        CvMat* dpdrot=NULL,
//        CvMat* dpdt=NULL,
//        CvMat* dpdf=NULL,
//        CvMat*dpdc=NULL,
//        CvMat* dpddist=NULL //偏导数的雅克比矩阵
//        )
        cvProjectPoints2( &pts_3, k == 0 ? _R1 : _R2, &Z, &A_tmp, 0, &pts );
        CvScalar avg = cvAvg(&pts);
        // double nx = imageSize.width, ny = imageSize.height 
        cc_new[k].x = (nx)/2 - avg.val[0];
        cc_new[k].y = (ny)/2 - avg.val[1];
        cout  << "cc_new["<<k<<"].x = "<< cc_new[k].x<< "\tnx = "<<nx<<"\tavg.val[0]=" <<avg.val[0]<<endl;
        cout  << "cc_new["<<k<<"].x = "<< cc_new[k].y<< "\tny = "<<nx<<"\tavg.val[1]=" <<avg.val[1]<<endl;
    }


    // vertical focal length must be the same for both images to keep the epipolar constraint
    // (for horizontal epipolar lines -- TBD: check for vertical epipolar lines)
    // use fy for fx also, for simplicity

    // For simplicity, set the principal points for both cameras to be the average
    // of the two principal points (either one of or both x- and y- coordinates)
    if( flags & CALIB_ZERO_DISPARITY )
    {   //为简单起见，将两个摄影机的主要点设置为平均值
        // cc_new[0].x=c1x , cc_new[0].y =c1y
        // cc_new[1].x=c2x , cc_new[1].y =c2y
        cc_new[0].x = cc_new[1].x = (cc_new[0].x + cc_new[1].x)*0.5;
        cc_new[0].y = cc_new[1].y = (cc_new[0].y + cc_new[1].y)*0.5;
    }
    else if( idx == 0 ) // horizontal stereo
        cc_new[0].y = cc_new[1].y = (cc_new[0].y + cc_new[1].y)*0.5;
    else // vertical stereo
        cc_new[0].x = cc_new[1].x = (cc_new[0].x + cc_new[1].x)*0.5;

    cvZero( &pp );
    //_pp = (mat)3X4 = P1
    //得到Ｐ1P2矩阵
    _pp[0][0] = _pp[1][1] = fc_new;
    _pp[0][2] = cc_new[0].x;
    _pp[1][2] = cc_new[0].y;
    _pp[2][2] = 1;
    cvConvert(&pp, _P1);

    _pp[0][2] = cc_new[1].x;
    _pp[1][2] = cc_new[1].y;
    _pp[idx][3] = _t[idx]*fc_new; // baseline * focal length
    cvConvert(&pp, _P2);
    // alpha-拉伸参数。如果设置为负或忽略，将不进行拉伸。如果设置为0，那么校正后图像只有有效的部分会被显示（没有黑色的部分），如果设置为1，那么就会显示整个图像。设置为0~1之间的某个值，其效果也居于两者之间。
    alpha = MIN(alpha, 1.); //double alpha=-1, 

    icvGetRectangles( _cameraMatrix1, _distCoeffs1, _R1, _P1, imageSize, inner1, outer1 );
    icvGetRectangles( _cameraMatrix2, _distCoeffs2, _R2, _P2, imageSize, inner2, outer2 );

    {
    newImgSize = newImgSize.width*newImgSize.height != 0 ? newImgSize : imageSize;
    double cx1_0 = cc_new[0].x;
    double cy1_0 = cc_new[0].y;
    double cx2_0 = cc_new[1].x;
    double cy2_0 = cc_new[1].y;
    double cx1 = newImgSize.width*cx1_0/imageSize.width;
    double cy1 = newImgSize.height*cy1_0/imageSize.height;
    double cx2 = newImgSize.width*cx2_0/imageSize.width;
    double cy2 = newImgSize.height*cy2_0/imageSize.height;
    double s = 1.;

    if( alpha >= 0 )
    {
        double s0 = std::max(std::max(std::max((double)cx1/(cx1_0 - inner1.x), (double)cy1/(cy1_0 - inner1.y)),
                            (double)(newImgSize.width - cx1)/(inner1.x + inner1.width - cx1_0)),
                        (double)(newImgSize.height - cy1)/(inner1.y + inner1.height - cy1_0));
        s0 = std::max(std::max(std::max(std::max((double)cx2/(cx2_0 - inner2.x), (double)cy2/(cy2_0 - inner2.y)),
                         (double)(newImgSize.width - cx2)/(inner2.x + inner2.width - cx2_0)),
                     (double)(newImgSize.height - cy2)/(inner2.y + inner2.height - cy2_0)),
                 s0);

        double s1 = std::min(std::min(std::min((double)cx1/(cx1_0 - outer1.x), (double)cy1/(cy1_0 - outer1.y)),
                            (double)(newImgSize.width - cx1)/(outer1.x + outer1.width - cx1_0)),
                        (double)(newImgSize.height - cy1)/(outer1.y + outer1.height - cy1_0));
        s1 = std::min(std::min(std::min(std::min((double)cx2/(cx2_0 - outer2.x), (double)cy2/(cy2_0 - outer2.y)),
                         (double)(newImgSize.width - cx2)/(outer2.x + outer2.width - cx2_0)),
                     (double)(newImgSize.height - cy2)/(outer2.y + outer2.height - cy2_0)),
                 s1);

        s = s0*(1 - alpha) + s1*alpha;
    }

    fc_new *= s;
    cout<<"S="<<s<<endl;
    cc_new[0] = cvPoint2D64f(cx1, cy1);
    cc_new[1] = cvPoint2D64f(cx2, cy2);

    cvmSet(_P1, 0, 0, fc_new);
    cvmSet(_P1, 1, 1, fc_new);
    cvmSet(_P1, 0, 2, cx1);
    cvmSet(_P1, 1, 2, cy1);

    cvmSet(_P2, 0, 0, fc_new);
    cvmSet(_P2, 1, 1, fc_new);
    cvmSet(_P2, 0, 2, cx2);
    cvmSet(_P2, 1, 2, cy2);
    cvmSet(_P2, idx, 3, s*cvmGet(_P2, idx, 3));

    if(roi1)
    {
        *roi1 = cv::Rect(cvCeil((inner1.x - cx1_0)*s + cx1),
                     cvCeil((inner1.y - cy1_0)*s + cy1),
                     cvFloor(inner1.width*s), cvFloor(inner1.height*s))
            & cv::Rect(0, 0, newImgSize.width, newImgSize.height);
    }

    if(roi2)
    {
        *roi2 = cv::Rect(cvCeil((inner2.x - cx2_0)*s + cx2),
                     cvCeil((inner2.y - cy2_0)*s + cy2),
                     cvFloor(inner2.width*s), cvFloor(inner2.height*s))
            & cv::Rect(0, 0, newImgSize.width, newImgSize.height);
    }
    }

    if( matQ )
    {
        double q[] =
        {
            1, 0, 0, -cc_new[0].x,
            0, 1, 0, -cc_new[0].y,
            0, 0, 0, fc_new,
            0, 0, -1./_t[idx],
            (idx == 0 ? cc_new[0].x - cc_new[1].x : cc_new[0].y - cc_new[1].y)/_t[idx]
        };
        CvMat Q = cvMat(4, 4, CV_64F, q);
        cvConvert( &Q, matQ );
    }
}



static void icvGetRectangles( const CvMat* cameraMatrix, const CvMat* distCoeffs,
                 const CvMat* R, const CvMat* newCameraMatrix, CvSize imgSize,
                 cv::Rect_<float>& inner, cv::Rect_<float>& outer )
{
    const int N = 9;
    int x, y, k;
    cv::Ptr<CvMat> _pts(cvCreateMat(1, N*N, CV_32FC2));
    CvPoint2D32f* pts = (CvPoint2D32f*)(_pts->data.ptr);

    for( y = k = 0; y < N; y++ )
        for( x = 0; x < N; x++ )
            pts[k++] = cvPoint2D32f((float)x*imgSize.width/(N-1),
                                    (float)y*imgSize.height/(N-1));

    cvUndistortPoints(_pts, _pts, cameraMatrix, distCoeffs, R, newCameraMatrix);

    float iX0=-FLT_MAX, iX1=FLT_MAX, iY0=-FLT_MAX, iY1=FLT_MAX;
    float oX0=FLT_MAX, oX1=-FLT_MAX, oY0=FLT_MAX, oY1=-FLT_MAX;
    // find the inscribed rectangle.
    // the code will likely not work with extreme rotation matrices (R) (>45%)
    for( y = k = 0; y < N; y++ )
        for( x = 0; x < N; x++ )
        {
            CvPoint2D32f p = pts[k++];
            oX0 = MIN(oX0, p.x);
            oX1 = MAX(oX1, p.x);
            oY0 = MIN(oY0, p.y);
            oY1 = MAX(oY1, p.y);

            if( x == 0 )
                iX0 = MAX(iX0, p.x);
            if( x == N-1 )
                iX1 = MIN(iX1, p.x);
            if( y == 0 )
                iY0 = MAX(iY0, p.y);
            if( y == N-1 )
                iY1 = MIN(iY1, p.y);
        }
    inner = cv::Rect_<float>(iX0, iY0, iX1-iX0, iY1-iY0);
    outer = cv::Rect_<float>(oX0, oY0, oX1-oX0, oY1-oY0);
}


