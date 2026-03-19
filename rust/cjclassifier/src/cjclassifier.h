/*
 * Copyright 2026 Jeremy Lilley (jeremy@jlilley.net)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CJCLASSIFIER_H
#define CJCLASSIFIER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a classifier instance. */
typedef struct CJClassifierHandle CJClassifierHandle;

/* Language codes returned by cj_detect / cj_detect_detailed. */
enum {
    CJ_UNKNOWN             = 0,
    CJ_CHINESE_SIMPLIFIED  = 1,
    CJ_CHINESE_TRADITIONAL = 2,
    CJ_JAPANESE            = 3,
};
typedef int32_t CJLanguageCode;

/* Detailed detection result. */
typedef struct {
    CJLanguageCode language;
    double gap;              /* 0 = dead heat, 1 = no contest */
    double total_scores[3];  /* [zh-hans, zh-hant, ja] */
} CJResult;

/* Load the bundled model. Returns NULL on failure. Free with cj_free(). */
CJClassifierHandle *cj_load_bundled(void);

/* Load a model from a file path. Returns NULL on failure. Free with cj_free(). */
CJClassifierHandle *cj_load_file(const char *path);

/* Detect language of a UTF-8 buffer. */
CJLanguageCode cj_detect(const CJClassifierHandle *model,
                          const char *text, size_t len);

/* Detect with detailed results. */
CJLanguageCode cj_detect_detailed(const CJClassifierHandle *model,
                                   const char *text, size_t len,
                                   CJResult *result);

/* Free a classifier handle (NULL-safe). */
void cj_free(CJClassifierHandle *model);

#ifdef __cplusplus
}
#endif

#endif /* CJCLASSIFIER_H */
