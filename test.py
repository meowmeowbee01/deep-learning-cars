import arcade

arcade.open_window(600, 600, "Drawing Example")

arcade.set_background_color([255,127,127])

arcade.start_render()

arcade.draw_lrtb_rectangle_filled(0, 300, 300, 0, [0,0,0])

arcade.finish_render()

arcade.run()
