#include <stdint.h>

int main(void)
{
    uint16_t adc_raw = 1023u;
    uint8_t adc_value = adc_raw;
    uint8_t samples[4] = {0u};

    samples[4] = adc_value;

    (void)adc_value;
    (void)samples;

    while (1)
    {
    }
}
