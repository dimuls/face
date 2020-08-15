package face

// #cgo pkg-config: dlib-1 opencv4
// #cgo CFLAGS: -Wall
// #cgo CXXFLAGS: -std=c++1z -Wall -O3 -DNDEBUG -march=native
// #cgo LDFLAGS: -lopencv_core -ldlib
// #include "recognizer.h"
import "C"

import (
	"errors"
	"image"
	"reflect"
	"unsafe"

	"gocv.io/x/gocv"
)

type Recognizer struct {
	recognizer unsafe.Pointer
}

func NewRecognizer(shaperModelFilePath, recognizerModelFilePath string) (*Recognizer, error) {
	cShaperModelFilePath := C.CString(shaperModelFilePath)
	defer C.free(unsafe.Pointer(cShaperModelFilePath))

	cRecognizerModelFilePath := C.CString(recognizerModelFilePath)
	defer C.free(unsafe.Pointer(cRecognizerModelFilePath))

	result := C.recognizer_init(cShaperModelFilePath, cRecognizerModelFilePath)
	defer C.free(unsafe.Pointer(result))

	if result.error_message != nil {
		defer C.free(unsafe.Pointer(result.error_message))
		return nil, errors.New(C.GoString(result.error_message))
	}

	return &Recognizer{recognizer: unsafe.Pointer(result.recognizer)}, nil
}

func (r *Recognizer) Close() {
	C.recognizer_free(r.recognizer)
	r.recognizer = nil
}

func (r *Recognizer) Recognize(img gocv.Mat, faceLocation image.Rectangle, padding float64, jittering int) (d Descriptor, err error) {
	cFaceLocation := C.rectangle{}
	cFaceLocation.min.x = C.int(faceLocation.Min.X)
	cFaceLocation.min.y = C.int(faceLocation.Min.Y)
	cFaceLocation.max.x = C.int(faceLocation.Max.X)
	cFaceLocation.max.y = C.int(faceLocation.Max.Y)

	result := C.recognizer_recognize(r.recognizer, unsafe.Pointer(img.Ptr()), &cFaceLocation, C.double(padding), C.int(jittering))
	defer C.free(unsafe.Pointer(result))

	if result.error_message != nil {
		defer C.free(unsafe.Pointer(result.error_message))
		err = errors.New(C.GoString(result.error_message))
		return
	}

	defer C.free(unsafe.Pointer(result.descriptor))

	var descriptor []C.float
	detectionsHeader := (*reflect.SliceHeader)(unsafe.Pointer(&descriptor))
	detectionsHeader.Cap = DescriptorSize
	detectionsHeader.Len = DescriptorSize
	detectionsHeader.Data = uintptr(unsafe.Pointer(result.descriptor))

	for i := range descriptor {
		d[i] = float32(descriptor[i])
	}

	return d, nil
}
