#include <stdlib.h>

#include "entity.h"

typedef struct {
    void* recognizer;
    char* error_message;
} recognizer_init_result;

typedef struct {
    float* descriptor;
    char* error_message;
} recognizer_recognize_result;

#ifdef __cplusplus
extern "C" {
#endif

recognizer_init_result* recognizer_init(char* shaper_model_file_path, char* recognizer_model_file_path);
void recognizer_free(void* recognizer);

recognizer_recognize_result* recognizer_recognize(void* recognizer, void* image, rectangle* face_location, double padding, int jittering);

#ifdef __cplusplus
}
#endif