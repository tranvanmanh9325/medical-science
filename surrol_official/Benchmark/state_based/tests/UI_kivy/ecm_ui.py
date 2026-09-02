ECM_panel_kv = '''MDBoxLayout:
    orientation: 'vertical'


    MDLabel:
        text: "                                                     Endoscope Fov Control"
        theme_text_color: "Primary"
        font_style: "H4"
        bold: True
        spacing: "10dp"
        size_hint: 1.0, 0.3


    MDSeparator:

    MDGridLayout:
        cols: 2
        spacing: "40dp"
        padding: "20dp", "10dp", "20dp", "20dp"
        size_hint: 1.0, 1.0
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/ecm_reach.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "ECM Reach"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Move the ECM to a randomly sampled position"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn1
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/misorient.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "MisOrient"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Adjust ECM to minimize misorientation"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn2
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"


        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/static_track.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Static Track"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Make ECM track a static target cube"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn3
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        

        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/active_track.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Active Track"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Make ECM track a active target cube"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0

                MDRaisedButton:
                    id: btn4
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"




    MDBoxLayout:
        MDRectangleFlatIconButton:
            icon: "exit-to-app"
            id: btn5
            text: "Exit"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.primary_color
            size_hint: 0.1, 0
        MDRectangleFlatIconButton:
            icon: "head-lightbulb-outline"
            id: btn6
            text: "AI Assistant"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.bg_light
            size_hint: 0.1, 0 
        MDRectangleFlatIconButton:
            icon: "chart-histogram"
            id: btn6
            text: "Evaluation"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.primary_color
            size_hint: 0.1,0
        MDRectangleFlatIconButton:
            icon: "help-box"
            id: btn6
            text: "Help"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.bg_light
            size_hint: 0.1,0


'''