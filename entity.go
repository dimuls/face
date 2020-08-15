package face

import "image"

type Detection struct {
	Rectangle  image.Rectangle
	Confidence float64
}

const DescriptorSize = 128

type Descriptor [DescriptorSize]float32