# Задание

Allocate GPU array arr of  float elements and initialize it with the kernel as follows: arr[i] = sin((i % 360) * Pi / 180). Copy array in CPU memory and count error as err = sum_i(abs(sin((i % 360) * Pi / 180) - arr[i]))/10^8. Investigate the dependence of the use of functions: sin, sinf, __sinf. Explain the result. Check the result for array of double data type.

# Результат

Средняя ошибка для различных тригонометрических функций:
| func | error | time, ms |
| -----| ----- | ---- |
| sin | 0.000000e+00 | 20.001860 |
| sinf | 4.362692e-08 | 7.744701 |
| __sinf | 1.276274e-07 | 7.688900 |

```Подробный вывод программы можно найти в stdout.txt```

Ошибки можно объяснить следующим образом:
- sin работает над числами двойной точности
- sinf работает над числами типа float
- __sinf это специальная intrinsic функция для быстрой аппроксимации значения синуса


