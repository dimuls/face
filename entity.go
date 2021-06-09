package face

import "image"

// Detection is a one face detection on image.
type Detection struct {
	Rectangle  image.Rectangle
	Confidence float64
}

// DescriptorSize is a face descriptor size.
const DescriptorSize = 128

// Descriptor is a face descriptor.
type Descriptor [DescriptorSize]float32