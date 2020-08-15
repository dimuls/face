package face

// #cgo pkg-config: dlib-1 opencv4
// #cgo CFLAGS: -Wall
// #cgo CXXFLAGS: -std=c++1z -Wall -O3 -DNDEBUG -march=native
// #cgo LDFLAGS: -lopencv_core -ldlib
// #include "detector.h"
import "C"

import (
	"errors"
	"image"
	"reflect"
	"unsafe"

	"gocv.io/x/gocv"
)

type Detector struct {
	detector unsafe.Pointer
}

func NewDetector(modelFilePath string) (*Detector, error) {
	cModelFilePath := C.CString(modelFilePath)
	defer C.free(unsafe.Pointer(cModelFilePath))

	result := C.detector_init(cModelFilePath)
	defer C.free(unsafe.Pointer(result))

	if result.error_message != nil {
		defer C.free(unsafe.Pointer(result.error_message))
		return nil, errors.New(C.GoString(result.error_message))
	}

	return &Detector{detector: unsafe.Pointer(result.detector)}, nil
}

func (d *Detector) Close() {
	C.detector_free(d.detector)
	d.detector = nil
}

func (d *Detector) Detect(img gocv.Mat) ([]Detection, error) {
	result := C.detector_detect(d.detector, unsafe.Pointer(img.Ptr()))
	defer C.free(unsafe.Pointer(result))

	if result.error_message != nil {
		defer C.free(unsafe.Pointer(result.error_message))
		return nil, errors.New(C.GoString(result.error_message))
	}

	defer C.free(unsafe.Pointer(result.detections))

	return convertCDetections(result.detections, result.detections_count), nil
}

func (d *Detector) BatchDetect(imgs []gocv.Mat) ([][]Detection, error) {
	if len(imgs) == 0 {
		return nil, nil
	}

	var imgPtrs []unsafe.Pointer
	var imgsRows, imgsCols int

	for i, img := range imgs {
		if i == 0 {
			imgsRows = img.Rows()
			imgsCols = img.Cols()
		} else if img.Rows() != imgsRows || imgsCols != img.Cols() {
			return nil, errors.New("images should have same size")
		}
		imgPtrs = append(imgPtrs, unsafe.Pointer(img.Ptr()))
	}

	result := C.detector_batch_detect(d.detector, unsafe.Pointer(&imgPtrs[0]), C.int(len(imgPtrs)))
	defer C.free(unsafe.Pointer(result))

	if result.error_message != nil {
		defer C.free(unsafe.Pointer(result.error_message))
		return nil, errors.New(C.GoString(result.error_message))
	}

	defer C.free(unsafe.Pointer(result.detections))

	var cBatchDetections *C.batch_detection = result.detections
	var batchDetections []C.batch_detection
	batchDetectionsHeader := (*reflect.SliceHeader)(unsafe.Pointer(&batchDetections))
	batchDetectionsHeader.Cap = int(result.detections_count)
	batchDetectionsHeader.Len = int(result.detections_count)
	batchDetectionsHeader.Data = uintptr(unsafe.Pointer(cBatchDetections))

	var bds [][]Detection

	for _, batchDetection := range batchDetections {
		bds = append(bds, convertCDetections(batchDetection.detections, batchDetection.detections_count))
	}

	return bds, nil
}

func convertCDetections(cDetections *C.detection, cDetectionsCount C.int) []Detection {

	var detections []C.detection
	detectionsHeader := (*reflect.SliceHeader)(unsafe.Pointer(&detections))
	detectionsHeader.Cap = int(cDetectionsCount)
	detectionsHeader.Len = int(cDetectionsCount)
	detectionsHeader.Data = uintptr(unsafe.Pointer(cDetections))

	var ds []Detection

	for _, detection := range detections {
		ds = append(ds, Detection{
			Rectangle: image.Rectangle{
				Min: image.Point{
					X: int(detection.region.min.x),
					Y: int(detection.region.min.y),
				},
				Max: image.Point{
					X: int(detection.region.max.x),
					Y: int(detection.region.max.y),
				},
			},
			Confidence: float64(detection.confidence),
		})
	}

	return ds
}
