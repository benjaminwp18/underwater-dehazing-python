# Underwater Image Enhancement

## TODO
 - UDCP
 - TCLAHE specify # blocks instead of $n=$ block size
 - Inefficient TCLAHE: image split into chunks twice (once for turbidity estimation & once for CLAHE)
 - Reimplement in C++
 - Contribute to OpenCV:
   - Dynamic-clip CLAHE
   - Error on writing wrongly sized frames to [VideoWriter](https://github.com/opencv/opencv/blob/09a85e97aa1aae4ec1a0d33edc04fcc36515aced/modules/videoio/src/cap.cpp#L666)
 - Smooth per-frame intermediate variables (consider calculating every $n$ frames)
 - Get better testing data
 - Set up other methods to compare with
 - Get metrics for quantitative evaluation