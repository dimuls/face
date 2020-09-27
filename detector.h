#include <stdlib.h>

#include "entity.h"

typedef struct
{
    void* detector;
    char* error_message;
} detector_init_result;

typedef struct
{
    detection* detections;
    int detections_count;
    char* error_message;
} detector_detect_result;

typedef struct
{
    batch_detection* detections;
    int detections_count;
    char* error_message;
} detector_batch_detect_result;

#ifdef __cplusplus
extern "C" {
#endif

detector_init_result* detector_init(char* model_file_path);
void detector_free(void* detector);

detector_detect_result* detector_detect(void* detector, void* image);
detector_batch_detect_result* detector_batch_detect(void* detector, void* images, int images_count);

#ifdef __cplusplus
}
#endif