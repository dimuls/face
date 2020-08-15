package main

import (
	"image"
	"image/color"
	"io/ioutil"
	"log"
	"math"
	"path"

	"gocv.io/x/gocv"

	"github.com/dimuls/face"
)

// Путь до папки с моделями. Папка должна содержать следующий файлы: dlib_face_recognition_resnet_model_v1.dat,
// mmod_human_face_detector.dat, shape_predictor_68_face_landmarks.dat. Архивы с этими файлами можно скачать из
// https://github.com/davisking/dlib-models.
const modelsPath = "./models"

// Путь до папки с персонами. Папка должна содержать на первом уровне папки, где назавнаие папки ― имя персоны,
// а втором уровне ― файлы с фотографиями лица соответствующей персоны.
const personsPath = "./persons"

// ID оборудования для получения видепотока. В нашем случае 0 ― это ID стандрантной веб-камеры.
const deviceID = 4

// Параметры векторизации, которые влияют на качество получаемого вектора:
const (
	padding   = 0.2 // насколько увеличивать квадрат выявленного лица;
	jittering = 30  // кол-во генерируемых немного сдвинутых и повёрнутых копий лица.
)

// Синий цвет.
var blue = color.RGBA{
	R: 0,
	G: 0,
	B: 255,
	A: 0,
}

// Минимальное расстояние соответствия персоне.
const matchDistance = 0.5

// Структура, описывающая персону.
type Person struct {
	// Имя персоны.
	Name string

	// Список дескрипторов лица персоны.
	Descriptors []face.Descriptor
}

func main() {
	// Инициализация детектора лиц, который будет выявлять лица.
	detector, err := face.NewDetector(path.Join(modelsPath, "mmod_human_face_detector.dat"))
	if err != nil {
		log.Fatalf("create detector: %v", err)
	}
	defer detector.Close()

	// Инициализация распознавателя лиц, который будет векторизовывать лица.
	recognizer, err := face.NewRecognizer(
		path.Join(modelsPath, "shape_predictor_68_face_landmarks.dat"),
		path.Join(modelsPath, "dlib_face_recognition_resnet_model_v1.dat"))
	if err != nil {
		log.Fatalf("create recognizer: %v", err)
	}
	defer recognizer.Close()

	// Инициализация базы персон.
	persons := loadPersons(detector, recognizer, personsPath)

	// Инициализация видеопотока.
	capture, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		log.Fatalf("open video capture: %v", err)
	}
	defer capture.Close()

	// Инициализация окна программы.
	window := gocv.NewWindow("face-recognizer")
	defer window.Close()

	// Инициализация изображения для очередного кадра.
	frame := gocv.NewMat()
	defer frame.Close()

	for {
		// Ждём 1 милисекунду нажатия клавиши на клавиатуре.
		// Если нажата Esc, то выходим из приложения.
		if window.WaitKey(1) == 27 {
			return
		}

		// Пока не получим кадр продолжаем цикл.
		if !capture.Read(&frame) {
			continue
		}

		// Выявляем лица в кадре.
		detects, err := detector.Detect(frame)
		if err != nil {
			log.Fatalf("detect faces: %v", err)
		}

		// Для каждого выявленного лица.
		for _, detect := range detects {

			// Получаем вектор выявленного лица.
			descriptor, err := recognizer.Recognize(frame, detect.Rectangle, padding, jittering)
			if err != nil {
				log.Fatalf("recognize face: %v", err)
			}

			// Ищем в массиве векторов известных лиц наиболее близкое (по евклиду) лицо.
			person, distance := findPerson(persons, descriptor)

			// Рисуем прямоугольник выявленного лица.
			gocv.Rectangle(&frame, detect.Rectangle, blue, 1)

			// Если расстояние между найденным известным лицом и выявленным лицом меньше
			// какого-то порога, то пишем имя найденного известного лица над нарисованным
			// прямоугольником.
			if distance <= matchDistance {
				gocv.PutText(&frame, person.Name, image.Point{
					X: detect.Rectangle.Min.X,
					Y: detect.Rectangle.Min.Y,
				}, gocv.FontHersheyComplex, 1, blue, 1)
			}
		}

		// Рисуем кадр в окне.
		window.IMShow(frame)
	}
}

// Вычисление евклидового расстояния.
func euclidianDistance(a, b face.Descriptor) float64 {
	var sum float64
	for i := range a {
		sum += math.Pow(float64(a[i])-float64(b[i]), 2)
	}
	return math.Sqrt(sum)
}

// Функция поиска наиболее близкой, заданному дескриптору, персоны.
func findPerson(persons []Person, descriptor face.Descriptor) (Person, float64) {
	// Объявляем переменные, которые будут хранить результаты поиска
	var minPerson Person
	var minDistance = math.MaxFloat64

	// Проходимся по каждой персоне.
	for _, person := range persons {
		// Проходимся по каждому дескриптору персоны.
		for _, personDescriptor := range person.Descriptors {
			// Вычисляем расстояние между дескриптором персоны и заданным дескриптором.
			distance := euclidianDistance(personDescriptor, descriptor)

			// Если полученное расстояние меньше текущего минимального, то сохраняем персону и расстояние в
			// переменные результатов.
			if distance < minDistance {
				minDistance = distance
				minPerson = person
			}
		}
	}
	return minPerson, minDistance
}

// Функция загрузки базы персон.
func loadPersons(detector *face.Detector, recognizer *face.Recognizer, personsPath string) (persons []Person) {
	// Читаем директорию, получаем массив его содержимого (информацию о файлах и папках).
	personsDirs, err := ioutil.ReadDir(personsPath)
	if err != nil {
		log.Fatalf("read persons directory: %v", err)
	}

	// По каждому элементу из директории персон.
	for _, personDir := range personsDirs {
		// Пропускаем не директории.
		if !personDir.IsDir() {
			continue
		}

		// Формируем персону.
		person := Person{
			Name: personDir.Name(), // Имя персоны ― название папки
		}

		// Читаем директорию персоны.
		personsFiles, err := ioutil.ReadDir(path.Join(personsPath, personDir.Name()))
		if err != nil {
			log.Fatalf("read person directory: %v", err)
		}

		// По каждому элементу из директории персоны.
		for _, personFile := range personsFiles {
			// Пропускаем если директория.
			if personFile.IsDir() {
				continue
			}

			filePath := path.Join(personsPath, personDir.Name(), personFile.Name())

			// Читаем и декодируем изображение
			img := gocv.IMRead(filePath, gocv.IMReadUnchanged)

			// Если не удалось прочитать файл и декодировать изображение, то пропускаем файл.
			if img.Empty() {
				continue
			}

			// Выявляем лица на изображении.
			detects, err := detector.Detect(img)
			if err != nil {
				log.Fatalf("detect on person image: %v", err)
			}

			// Если кол-во лиц не 1, то завершаем программу с ошибкой.
			if len(detects) != 1 {
				log.Fatalf("multple faces detected on photo %s", filePath)
			}

			// Получаем вектор лица на изображении.
			descriptor, err := recognizer.Recognize(img, detects[0].Rectangle, padding, jittering)
			if err != nil {
				log.Fatalf("recognize persons face: %v", err)
			}

			// Добавляем вектор в массив векторов персоны.
			person.Descriptors = append(person.Descriptors, descriptor)

			img.Close()
		}

		// Добавляем очередную персону в массив перосн.
		persons = append(persons, person)
	}

	return persons
}
