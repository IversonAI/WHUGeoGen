# 给定的图像编号信息
zoom_level = 16
x =24625
y = 19317

# 计算从 16 级别到 15 级别的转换
new_zoom_level_15 = 15
# new_x_15 = x // 2
# new_y_15 = y // 2

new_x_15 = x // 2
new_y_15 = y // 2
# 16_19320_24626

# 通过新的 x、y 值和 15 级别的 zoom level 计算新的图像编号
new_image_number_15 = f"{new_zoom_level_15}_{new_x_15}_{new_y_15}"

print("15 级别的图像编号是:", new_image_number_15)
# 15_12308_9654


# 24615_19305
# 计算从 16 级别到 17 级别的转换
new_zoom_level_17 = 17
new_x_17 = x * 2
new_y_17 = y * 2



# 通过新的 x、y 值和 17 级别的 zoom level 计算新的图像编号
new_image_number_17 = f"{new_zoom_level_17}_{new_x_17}_{new_y_17}"

print("17 级别的图像编号是:", new_image_number_17)
# 17_49230_38610

# 计算从 17 级别到 18 级别的转换
new_zoom_level_18 = 18
new_x_18 = new_x_17 * 2
new_y_18 = new_y_17 * 2

# 通过新的 x、y 值和 18 级别的 zoom level 计算新的图像编号
new_image_number_18 = f"{new_zoom_level_18}_{new_x_18}_{new_y_18}"

print("18 级别的图像编号是:", new_image_number_18)
# 18_98460_77220

# 计算从 18 级别到 19 级别的转换
new_zoom_level_19 = 19
new_x_19 = new_x_18 * 2
new_y_19 = new_y_18 * 2


# 通过新的 x、y 值和 19 级别的 zoom level 计算新的图像编号
new_image_number_19 = f"{new_zoom_level_19}_{new_x_19}_{new_y_19}"


print("19 级别的图像编号是:", new_image_number_19)
# 19_196920_154440


# 计算从 19 级别到 20 级别的转换
new_zoom_level_20 = 20
new_x_20 = new_x_19 * 2
new_y_20 = new_y_19 * 2


# 通过新的 x、y 值和 20 级别的 zoom level 计算新的图像编号
new_image_number_20 = f"{new_zoom_level_20}_{new_x_20}_{new_y_20}"


print("20 级别的图像编号是:", new_image_number_20)
# 20_393840_308880
