import os
import sys
import argparse
import subprocess
from pathlib import Path

# Ensure proper path setup
current_script = Path(__file__).resolve()
marmo3d_dir = current_script.parent.parent  # src/marmo3d/
src_dir = marmo3d_dir.parent                # src/
project_root = src_dir.parent               # project root

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
import marmo3d.marmo_dataset as marmo_dataset
from marmo3d.blender import render_setting
from marmo3d.blender import render_csv
from marmo3d.blender import render_spec

def main(profile, frames, **kwargs):
    
    # 変数
    ## kwargs
    split = kwargs.get('split')
    overrides = kwargs.get('overrides')
    workers = kwargs.get('workers')
    dry_run = kwargs.get('dry_run')
    ## 
    project_root = Path(".").resolve()
    dataset_spec_path = project_root / "config" / 'dataset_spec.yaml'
    render_spec_path = Path(project_root) / "config" / "render_spec.yaml"
    if not render_spec_path.exists():
        render_spec_path = marmo3d_dir / "config" / "render_spec.yaml"
    
    dataset_info = marmo_dataset.MarmoDataset.from_yaml(dataset_spec_path, project_root=project_root)
    render_info = render_spec.RenderSpec.from_yaml(render_spec_path) # ,project_root=project_root
    

    sessions = dataset_info.iter_sessions(valid_only=True, log_skips=True)
    
    # TODO render_spec.yamlの情報とdataset_yamlの情報を分けるかどうか
    # TODO 分けるとしたらどうするの
    
    # 出力ログ
    print("base.blend:", render_info.resolve_base_blend(project_root))
    print(type(render_info.resolve_base_blend(project_root)))
    print("resolution:", render_info.get_resolution(project_root))
    views = render_info.get_views(project_root)
    print("views:", [v.view_id for v in views])
    print("frames:", frames)

    # TODO 
    # {session: dailylife, frames: [100, 235]},
    # {session: dailylife, frames: [100, 235]},
    # {session: dailylife, frames: [100, 235]},
    for s in sessions:
        animal = s.animal
        session_id = s.session_id
        external_pickle_path = s.kp3d_path()

        session_path = os.path.join(dataset_info.results_root, dataset_info.results_subdir, session_id)
        req = render_setting.RenderRequest( 
            # TODO ここ以降にreqとrender_specが混在してて気持ち悪い
            # TODO RenderRequestって本当にいる？
            profile=profile,
            sessions=session_path,
            frames=frames,
            split=split,
            overrides=overrides,
            workers=workers,
            dry_run=dry_run,
        )

        render_setting.CheckRequest().validate(req, render_info)

        # Use an absolute path so Blender doesn't resolve it relative to its install dir.
        output_root = project_root / "outputs" / "validation" / "render"
        plan = render_setting.RenderPlan.from_request(
            req,
            output_dir_resolver=lambda session, _frames: output_root / session.name,
        )

        # CSV結果を構築（再利用可能な関数を使用）
        csv_results = render_csv.build_csv_results(
            plan_items=plan.items,
            kp3d_finder=render_setting._find_kp3d_pickle,
            render_spec_path=Path(render_spec_path),
            output_root=output_root,
            request_all_frames=(req.frames is None),
        )

        base_blend = render_info.resolve_base_blend(project_root)
        print("Base blend path:", base_blend)
        print("Render spec path:", render_spec_path)
        print("Output root:", output_root)
        print("Render plan items:", len(plan.items))
        script_path = Path(__file__).resolve().parents[2] / "class_blueprint/blender/render.py"
        commands = render_setting.RenderJob(
            base_blend=base_blend, # TODO render_spec
            render_spec_path=render_spec_path,
            script_path=script_path, # TODO render_specで指定
            output_root=output_root, # TODO render_specで指定
            profile=req.profile,
        ).make_commands(
            plan,
            extra_args_resolver=lambda item: [
                "--csv_path",
                str(csv_results[(str(item.session), tuple(item.frames))]["csv_path"]),
                "--all_frames",
                "1" if csv_results[(str(item.session), tuple(item.frames))]["all_frames"] else "0",
            ]
            if (str(item.session), tuple(item.frames)) in csv_results
            else [],
        )
        print("Generated {} commands.".format(len(commands)))
        render_setting.Executor(workers=req.workers, retries=1, timeout=None).execute(
            commands, dry_run=req.dry_run
        )
        
        # manifest（例: image）
        # TODO Executor実行時にRenderManifestを実行するようにする
        # TODO　ここにこれを記述しない
            # m = render_spec.RenderManifest(
            #     animal=animal,
            #     session_id=session_id,
            #     base_blend_path=str(render_info.resolve_base_blend(project_root)),
            #     render_profile=render_info.profile,
            #     view_ids=[v.view_id for v in views],
            #     frames=frames,
            #     kp3d_source_path=str(render_info.temp3d_path(project_root, session_id, frames[0])),
            #     status="success",
            # )
            # m.save_json(render_info.manifest_path(project_root, animal, session_id))
        exit()


if __name__ == "__main__":
    profile = 'env_camera'
    frames = [50, 100, 150]
    main(profile, 
         frames,
         split=2,
         overrides=True,
         workers=2,
         dry_run=False,
         )
