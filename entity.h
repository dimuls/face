#pragma once

typedef struct {
    int x;
    int y;
} point;

typedef struct {
    point min;
    point max;
} rectangle;

typedef struct {
    rectangle region;
    double confidence;
} detection;

typedef struct {
    detection* detections;
    int detections_count;
} batch_detection;