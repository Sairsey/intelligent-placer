# intelligent-placer
Лабораторная работа по написанию Intelligent Placer-а, который по поданной на вход фотографии нескольких предметов на светлой горизонтальной поверхности и многоугольнику понимать, можно ли расположить одновременно все эти предметы на плоскости так, чтобы они влезли в этот многоугольник.
# Постановка задачи
### Общие
- Программа получает на вход путь до изображения со всеми объектами и нарисованным на листе многоугольником
- Программа должна выдать ответ хотя бы за 15 минут
- Ответом может быть либо True либо False
- Ответ выводится в стандартный поток вывода

### Ограничения по задаче
- Многоугольник должен быть замкнут 
- Многоугольник должен быть выпуклым
- Один объект присутствовать на фото только один раз
- Программа отвечает False если входные данные некорректные
- Программа расценивает любые снятые объекты как абсолютно твёрдые тела
- Дыры внутри объектов не учитываются при работе программы 

### Содержание фотографий
- Объекты и многоугольник должны быть размещены на белой поверхности
- Объекты рассматриваются только из тренировочного набора данных
- Объекты не должны пересекаться друг с другом
- Объекты не должны пересекаться с многоугольником
- Объекты и многоугольник должны целиком помещаться на фото
- Объекты должны распологаться вне многоугольника
- Многоугольник задаётся черным маркером на белом листе бумаги.

### Фотометрические
- фотографии должны быть в формате *\*.jpg*
- Угол между направлением камеры и перпендикуляром к поверхности должен быть не более *15&deg;*
- Освещение и цвета объектов должны быть подобраны так, чтобы они были легко отличимы от поверхности

# Изображения использованных объектов
[Тыц](images)

# Тесты
[Тыц](test_cases)