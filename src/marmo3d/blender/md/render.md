in run_render.py
        commands = render_setting.RenderJob(
            base_blend=base_blend,
            render_spec_path=render_spec_path,
            script_path=Path("class_blueprint/blender/render.py"),
            output_root=Path("./render"),
            profile=req.profile,
        ).make_commands(plan)

コマンドをこれで整えました

しかし、render.pyで実行しているのですがかなり出来損ないです
そこで、render.pyを完全に新しく書き換えるか修正するかしたいです

全てのフレームを実行していないのにfullが入ってる
あと100と200フレームを指定したはずなのにcsvには100しか入ってない
しかも上の指定で100フレーム分armatureにキーフレームが挿入されている
blenderファイルの100と200フレームにキーフレームが挿入されていることが求められている
あと、manifestに同じ名前が複数ある
100と200でそれぞれのカメラの名前が入っているのだとおもう
セッションの中の情報なので別々のカメラが使われているのならまだしも同じカメラを使っているのなら冗長なので重複は省くこと
blenderの名前が悪い"--job_id", f"{args.session_id}_full"　run_render.pyの中のここが悪い
frameだけ指定されているなら名前を変えたほうがいい
全フレーム入れているならall_frameとかにするべき
manifestファイルはrenderingしたあとの画像フォルダに入るべき
csvは画像フォルダに入れられない
それはcsvを使ってblenderにアニメーションを入れ込んでいる
そのときに複数フレームのcsvの情報がある
そして複数のキーフレームをアニメーションとして設定する
それをする