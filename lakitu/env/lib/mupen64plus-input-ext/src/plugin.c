#include <stddef.h>
#include <string.h>

#include "m64p_common.h"
#include "m64p_plugin.h"
#include "m64p_types.h"

#define PLUGIN_NAME              "Mupen64Plus Input Extension Plugin"
#define PLUGIN_VERSION            0x020600
#define INPUT_PLUGIN_API_VERSION  0x020100
#define CONFIG_API_VERSION        0x020100

typedef struct _m64p_input_extension_functions
{
	ptr_GetKeys             InputExtFuncGetKeys;
	ptr_InitiateControllers InputExtFuncInitiateControllers;
	ptr_RenderCallback      InputExtFuncRenderCallback;
} m64p_input_extension_functions;

static m64p_input_extension_functions l_ExternalInputFuncTable = {NULL, NULL, NULL};
static int l_PluginInit = 0;
static int l_InputExtensionActive = 0;


EXPORT m64p_error CALL PluginStartup(m64p_dynlib_handle handle, void *context, void (*callback)(void *, int, const char *)) {
    if (l_PluginInit)
        return M64ERR_ALREADY_INIT;

    l_PluginInit = 1;
    memset(&l_ExternalInputFuncTable, 0, sizeof(m64p_input_extension_functions));
    return M64ERR_SUCCESS;
}

EXPORT m64p_error CALL PluginShutdown(void) {
    if (!l_PluginInit)
        return M64ERR_NOT_INIT;

    l_PluginInit = 0;
    memset(&l_ExternalInputFuncTable, 0, sizeof(m64p_input_extension_functions));
    return M64ERR_SUCCESS;
}

EXPORT m64p_error CALL PluginGetVersion(m64p_plugin_type *PluginType, int *PluginVersion, int *APIVersion, const char **PluginNamePtr, int *Capabilities) {
    if (PluginType != NULL)
        *PluginType = M64PLUGIN_INPUT;
    if (PluginVersion != NULL)
        *PluginVersion = PLUGIN_VERSION;
    if (APIVersion != NULL)
        *APIVersion = INPUT_PLUGIN_API_VERSION;
    if (PluginNamePtr != NULL)
        *PluginNamePtr = PLUGIN_NAME;
    if (Capabilities != NULL)
        *Capabilities = 0;
    return M64ERR_SUCCESS;
}

EXPORT m64p_error CALL OverrideInputExt(m64p_input_extension_functions *InputFunctionStruct) {
    if (InputFunctionStruct == NULL)
        return M64ERR_INPUT_ASSERT;

    if (InputFunctionStruct->InputExtFuncGetKeys == NULL ||
        InputFunctionStruct->InputExtFuncInitiateControllers == NULL ||
        InputFunctionStruct->InputExtFuncRenderCallback == NULL)
        return M64ERR_INPUT_ASSERT;

    memcpy(&l_ExternalInputFuncTable, InputFunctionStruct, sizeof(m64p_input_extension_functions));
    l_InputExtensionActive = 1;
    return M64ERR_SUCCESS;
}

EXPORT void CALL GetKeys(int Control, BUTTONS *Keys) {
    if (l_InputExtensionActive)
        l_ExternalInputFuncTable.InputExtFuncGetKeys(Control, Keys);
}
EXPORT void CALL InitiateControllers(CONTROL_INFO ControlInfo) {
    if (l_InputExtensionActive)
        l_ExternalInputFuncTable.InputExtFuncInitiateControllers(ControlInfo);
}
EXPORT void CALL RenderCallback(void) {
    if (l_InputExtensionActive)
        l_ExternalInputFuncTable.InputExtFuncRenderCallback();
}

EXPORT void CALL ControllerCommand(int Control, unsigned char *Command) {}
EXPORT void CALL ReadController(int Control, unsigned char *Command) {}
EXPORT void CALL RomClosed(void) {}
EXPORT void CALL RomOpen(void) {}
EXPORT void CALL SDL_KeyDown(int keymod, int keysym) {}
EXPORT void CALL SDL_KeyUp(int keymod, int keysym) {}
EXPORT void CALL SendVRUWord(uint16_t length, uint16_t *word, uint8_t lang) {}
EXPORT void CALL SetMicState(int state) {}
EXPORT void CALL ReadVRUResults(uint16_t *error_flags, uint16_t *num_results, uint16_t *mic_level, uint16_t *voice_level, uint16_t *voice_length, uint16_t *matches) {}
EXPORT void CALL ClearVRUWords(uint8_t length) {}
EXPORT void CALL SetVRUWordMask(uint8_t length, uint8_t *mask) {}
